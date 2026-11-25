# src/gpt/minigpt_selector.py
"""MiniGPT 근거 선택기 (서비스 연결용, 실험 기능)

search.answer(q, mode, selector=select)로 넘기면 규칙 기반 rule_select 대신 쓰인다.
역할: 검색된 후보 줄마다 P(이 줄이 질문의 답)를 계산해 가장 높은 줄을 고른다. 답변 문장은 만들지 않는다.

- 학습한 문항 종류(스킬·패치 변경·기본 능력치)이고 후보가 2개 이상일 때만 MiniGPT가 고른다.
  그 밖(아이템·템트리·역할군 등, 또는 후보 1개)은 규칙 기반 결과를 그대로 쓴다.
- 서비스 줄 표기를 학습 데이터 표기로 맞춘다 ('26.19 패치' -> '26-19 패치', '⇒ ' -> '⇒').
- 최고 점수가 0.5 미만이어도 '근거 없음'으로 끊지 않고 가장 높은 줄을 고른다 (서비스 응답이 사라지지 않게).
  점수는 scores()로 따로 받아 화면에 보여 준다.
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

GPT_DIR = Path(__file__).resolve().parent
ROOT = GPT_DIR.parents[1]
sys.path.insert(0, str(GPT_DIR))

CKPT = ROOT / "checkpoints" / "minigpt_sft8.pt"    # 서비스에 쓰는 체크포인트 (실험 8, 4지선다 76.0%)
LINE_CHARS = 80                                     # 학습 때와 같은 근거 줄 길이
INTENTS = {"skill", "patch", "stats"}               # MiniGPT가 학습한 문항 종류
PATCH_RE = re.compile(r"(\d{2})\.(S?\d+(?:\.\d+)?) 패치")


@lru_cache(maxsize=1)
def _load():
    import torch
    from lol_tokenizer import load as load_tokenizer, format_chat
    from mini_gpt import MiniGPT

    tok = load_tokenizer(str(ROOT / "tokenizer_out"))
    ck = torch.load(CKPT, map_location="cpu")
    cfg = {k: v for k, v in ck["config"].items() if k != "dropout"}
    model = MiniGPT(**cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    end_id = tok.convert_tokens_to_ids("<|end|>")
    ans = {}
    for i in (0, 1):
        ids = tok.encode(f"{i}<|end|>", add_special_tokens=False)
        assert len(ids) == 2 and ids[1] == end_id
        ans[i] = ids[0]
    return torch, tok, format_chat, model, ans[0], ans[1], cfg["block_size"]


def to_train_format(text):
    """서비스 근거 줄 -> 학습 데이터 표기"""
    line = text.split("\n", 1)[0]
    line = PATCH_RE.sub(lambda m: f"{m.group(1)}-{m.group(2).replace('.', '-')} 패치", line)
    line = line.replace("⇒ ", "⇒")
    return line if len(line) <= LINE_CHARS else line[:LINE_CHARS] + "…"


def scores(q, texts):
    """질문과 후보 줄들 -> 줄마다 P(일치) 리스트"""
    if not texts:
        return []
    torch, tok, format_chat, model, no_id, yes_id, block = _load()
    prompts = []
    for t in texts:
        p = format_chat([{"role": "system", "content": f"근거: {to_train_format(t)}"},
                         {"role": "user", "content": q}], add_generation_prompt=True)
        prompts.append(tok.encode(p, add_special_tokens=False)[-block:])
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            b = prompts[i:i + 32]
            L = max(len(p) for p in b)
            x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            for j, p in enumerate(b):
                x[j, :len(p)] = torch.tensor(p)
            last = torch.tensor([len(p) - 1 for p in b])
            lg = model(x)[torch.arange(len(b)), last][:, [no_id, yes_id]]
            out += torch.softmax(lg.float(), dim=-1)[:, 1].tolist()
    return out


def applies(P, hits):
    """스킬 질문은 슬롯이 있을 때만 (슬롯 없는 질문으로는 학습하지 않음)"""
    if P.intent == "skill" and not P.slot:
        return False
    return P.intent in INTENTS and len(hits) >= 2


def model_query(P):
    """줄임말('마이 궁')로 물으면 줄에 적힌 정식 이름이 질문에 없으므로 앞에 붙여 준다"""
    if P.champion and P.champion not in P.query:
        return f"{P.champion} {P.query}"
    return P.query


def hit_scores(P, hits):
    return scores(model_query(P), [h["text"] for h in hits])


def select(P, hits):
    """search.answer()의 selector 자리에 넣는 함수: hits 인덱스를 돌려준다"""
    import search
    if not applies(P, hits):
        return search.rule_select(P, hits)
    ps = hit_scores(P, hits)
    return max(range(len(ps)), key=lambda i: ps[i])
