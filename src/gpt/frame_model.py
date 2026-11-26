"""실험 11: MiniGPT 질문 해석 모델 — 공용 부품 (학습·평가·서비스가 같이 씀)

설계: src/gpt/experiments/exp11_frame_spec.md
  - 이름 찾기는 사전: 챔피언·아이템 이름과 패치 번호를 번호 붙은 자리표시([챔피언1], [아이템1], [패치1])로 바꾼다.
  - 입력 = "이전: <이전 틀, 자리표시로> | 질문: <자리표시로 바꾼 질문>"
  - 출력 = 필드별 분류 머리 (생성 X): 문장 머리(마지막 토큰) + 챔피언 자리표시마다 역할 머리(그 자리표시 끝 토큰)
"""
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "gpt"))
sys.path.insert(0, str(ROOT / "src" / "rag"))

from mini_gpt import MiniGPT  # noqa: E402

# ---------------- 필드와 값 (exp11_frame_spec.md 3장) ----------------
INTENTS = ["개요", "스킬", "패치", "이유", "능력치", "템트리", "상대법", "성능", "아이템", "추천", "안내"]
HEADS = {   # 필드 -> 값 목록 (None = 비어 있음). 아이템·패치 자리표시는 사전이 찾으므로 '있다/없다'만 정함
    "범위": ["롤", "사이트밖", "데이터밖"],
    "스킬": [None, "패시브", "Q", "W", "E", "R", "전체"],
    "라인": [None, "탑", "정글", "미드", "원딜", "서폿"],
    "역할군": [None, "탱커", "전사", "마법사", "암살자", "원딜", "서포터"],
    "난이도": [None, "쉬움"],
    "시점": [None, "최신", "이력", "패치"],          # '패치' = 질문에 적힌 [패치k]
    "방향": [None, "상향", "하향", "변경"],
    "개수": [None, "하나", "전체"],
    "이어받음": [False, True],
}
ROLES = ["관계없음", "나", "상대", "대상"]
PREV_KEYS = ["라인", "스킬", "역할군", "난이도", "시점", "방향", "개수"]


# ---------------- 사전: 이름 -> 자리표시 ----------------
class Masker:
    def __init__(self):
        import search as S
        self.S = S
        terms = {}
        for n in S.NAMES:
            terms[n] = ("챔피언", n)
        for n in S.ITEM_NAMES:
            terms.setdefault(n, ("아이템", n))
        for a, n in S.ITEM_ALIASES.items():
            terms.setdefault(a, ("아이템", n))
        self.terms = terms
        self.re = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)))

    def mask(self, q, table=None):
        """q의 이름을 자리표시로. table: {(종류, 정식이름): 자리표시}를 이어서 씀(이전 틀과 번호 공유)"""
        table = {} if table is None else table
        q = self.S.normalize(q)

        def ph(kind, name):
            key = (kind, name)
            if key not in table:
                k = sum(1 for t in table if t[0] == kind) + 1
                table[key] = f"[{kind}{k}]"
            return table[key]

        q = self.S.PATCH_RE.sub(lambda m: ph("패치", m[0].replace(" ", "")), q)
        q = self.re.sub(lambda m: ph(*self.terms[m[0]]), q)
        return q, table

    def prev_text(self, frame, table):
        """이전 틀 -> '의도=템트리 나=[챔피언1] 상대=[챔피언2] 라인=탑' (자리표시 번호를 table에 등록)"""
        if not frame:
            return "없음"
        parts = []
        if frame.get("범위") and frame["범위"] != "롤":
            parts.append(f"범위={frame['범위']}")
        if frame.get("의도"):
            parts.append("의도=" + ",".join(frame["의도"]))
        for c in frame.get("챔피언") or []:
            if c["역할"] in ("나", "상대", "대상"):
                key = ("챔피언", c["이름"])
                if key not in table:
                    table[key] = f"[챔피언{sum(1 for t in table if t[0] == '챔피언') + 1}]"
                parts.append(f"{c['역할']}={table[key]}")
        if frame.get("아이템"):
            key = ("아이템", frame["아이템"])
            if key not in table:
                table[key] = f"[아이템{sum(1 for t in table if t[0] == '아이템') + 1}]"
            parts.append(f"아이템={table[key]}")
        for k in PREV_KEYS:
            v = frame.get(k)
            if v:
                parts.append(f"{k}={v}")
        return " ".join(parts) or "없음"


# ---------------- 토큰화: 자리표시 위치를 정확히 알기 위해 조각별로 인코딩 ----------------
PH_RE = re.compile(r"\[(챔피언|아이템|패치)\d+\]")


def encode(tok, prev_txt, q_masked, block):
    """-> (토큰 ids, {자리표시: 그 자리표시 끝 토큰 위치(질문 쪽 마지막 등장)})"""
    head = tok.encode(f"<|system|>질문 해석<|end|><|user|>이전: {prev_txt} | 질문: ", add_special_tokens=False)
    ids, pos = list(head), {}
    # 이전 쪽 자리표시도 위치를 기록 (질문에 다시 안 나오는 '이어받은' 챔피언용)
    for m in PH_RE.finditer(prev_txt):
        pre = tok.encode(f"<|system|>질문 해석<|end|><|user|>이전: {prev_txt[:m.end()]}", add_special_tokens=False)
        pos[m[0]] = len(pre) - 1
    i = 0
    for m in PH_RE.finditer(q_masked):
        if m.start() > i:
            ids += tok.encode(q_masked[i:m.start()], add_special_tokens=False)
        ids += tok.encode(m[0], add_special_tokens=False)
        pos[m[0]] = len(ids) - 1
        i = m.end()
    if i < len(q_masked):
        ids += tok.encode(q_masked[i:], add_special_tokens=False)
    ids += tok.encode("<|end|><|assistant|>", add_special_tokens=False)
    if len(ids) > block:                          # 너무 길면 앞(이전 틀 쪽)을 자르지 않고 질문 끝을 자른다
        tail = tok.encode("<|end|><|assistant|>", add_special_tokens=False)
        ids = ids[:block - len(tail)] + tail
        pos = {k: v for k, v in pos.items() if v < block - len(tail)}
    return ids, pos


# ---------------- 모델: MiniGPT 몸통 + 분류 머리 ----------------
class FrameModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=6, n_head=8, block_size=192, dropout=0.1):
        super().__init__()
        self.gpt = MiniGPT(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head,
                           block_size=block_size, dropout=dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(d_model, len(v)) for k, v in HEADS.items()})
        self.intent = nn.Linear(d_model, len(INTENTS))           # 의도마다 예/아니오 (여러 개 가능)
        self.first = nn.Linear(d_model, len(INTENTS))            # 가장 중요한 의도 하나 (답 모듈 순서)
        self.role = nn.Linear(2 * d_model, len(ROLES))           # [자리표시 위치, 문장 끝] 이어 붙여 판단

    def hidden(self, idx):
        g = self.gpt
        T = idx.size(1)
        x = g.drop(g.transformer.wte(idx) + g.transformer.wpe(torch.arange(T, device=idx.device)))
        for b in g.transformer.h:
            x = b(x)
        return g.transformer.ln_f(x)

    def forward(self, idx, last, ph_index):
        """idx (B,T), last (B,) 문장 끝 위치, ph_index (N,2) = [배치 번호, 자리표시 위치]"""
        h = self.hidden(idx)
        B = idx.size(0)
        s = h[torch.arange(B, device=idx.device), last]
        out = {k: head(s) for k, head in self.heads.items()}
        out["의도"] = self.intent(s)
        out["첫의도"] = self.first(s)
        if ph_index.numel():
            out["역할"] = self.role(torch.cat([h[ph_index[:, 0], ph_index[:, 1]], s[ph_index[:, 0]]], -1))
        else:
            out["역할"] = s.new_zeros((0, len(ROLES)))
        return out


# ---------------- 학습 예시 만들기 / 틀 되돌리기 ----------------
def to_example(masker, tok, row, block):
    """{'이전': {'틀'} | None, '질문', '틀'} -> 모델 입력·정답. 틀의 챔피언이 자리표시로 안 잡히면 None (자동 검증)"""
    table = {}
    prev = row.get("이전")
    prev_txt = masker.prev_text(prev and prev.get("틀"), table)
    qm, table = masker.mask(row["질문"], table)
    ids, pos = encode(tok, prev_txt, qm, block)
    f = row["틀"]
    name_of = {v: k[1] for k, v in table.items() if k[0] == "챔피언"}
    gold_role = {c["이름"]: c["역할"] for c in f.get("챔피언") or []}
    if any(("챔피언", n) not in table for n in gold_role):
        return None                                   # 틀의 챔피언이 질문·이전 틀 어디에도 없음
    roles = [(p, pos[p], ROLES.index(gold_role.get(name_of[p], "관계없음"))) for p in name_of if p in pos]
    y = {}
    for k, vals in HEADS.items():
        v = f.get(k)
        if k == "시점" and v not in vals:             # 실제 패치 번호 -> '패치'
            v = "패치"
        if v not in vals:
            return None
        y[k] = vals.index(v)
    y["의도"] = [1.0 if i in (f.get("의도") or []) else 0.0 for i in INTENTS]
    order = f.get("의도") or []
    y["첫의도"] = INTENTS.index(order[0]) if order else -100      # 의도가 없으면(사이트밖 등) 학습 제외
    return {"ids": ids, "roles": roles, "y": y, "order": order, "table": table}


def decode(out_one, roles_one, ex, intent_thr=0.5):
    """모델 출력(한 문장) -> 틀 (실제 이름). roles_one = [(자리표시, 역할 로짓)]"""
    f = {k: HEADS[k][int(out_one[k].argmax())] for k in HEADS}
    p = torch.sigmoid(out_one["의도"])
    first = int(out_one["첫의도"].argmax())
    rest = [i for i in p.argsort(descending=True).tolist() if p[i] >= intent_thr and i != first][:2]
    f["의도"] = [INTENTS[i] for i in [first] + rest] if f["범위"] == "롤" else []
    name_of = {v: k[1] for k, v in ex["table"].items() if k[0] == "챔피언"}
    f["챔피언"] = [{"이름": name_of[ph], "역할": ROLES[int(r.argmax())]}
                  for ph, r in roles_one if ROLES[int(r.argmax())] != "관계없음"]   # 관계없음은 학습용 라벨, 틀에는 안 씀
    items = [k[1] for k, v in ex["table"].items() if k[0] == "아이템"]
    f["아이템"] = items[0] if items and "아이템" in f["의도"] else None
    if f["시점"] == "패치":
        pts = [k[1] for k, v in ex["table"].items() if k[0] == "패치"]
        f["시점"] = pts[0] if pts else "최신"
    return f
