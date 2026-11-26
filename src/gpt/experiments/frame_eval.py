"""실험 11 평가: 해석 틀(exp11_frame_spec.md) 정답 세트로 '틀을 내는 함수'를 채점한다.

정답 세트: logs/eval/exp11_real_gold.jsonl (11주차 실제 질문, 공개 저장소라 gitignore된 logs/에 둠)
  한 줄 = {"질문", "이전": {"질문", "틀"} | null, "틀": 정답, ...}
채점 대상: predict(질문, 이전틀) -> 틀(dict). 지금은 규칙 기준선(rule_frame)만 있고, MiniGPT 모델이 생기면 같은 함수로 채점.

실행 (프로젝트 루트):  python3.10 src/gpt/experiments/frame_eval.py [--show]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rag"))
GOLD = ROOT / "logs" / "eval" / "exp11_real_gold.jsonl"

SCALAR = ["범위", "아이템", "스킬", "라인", "역할군", "난이도", "시점", "방향", "개수", "이어받음"]
EMPTY = {"범위": "롤", "의도": [], "챔피언": [], "아이템": None, "스킬": None, "라인": None, "역할군": None,
         "난이도": None, "시점": None, "방향": None, "개수": None, "이어받음": False}


# ---------------- 규칙 기준선: search.parse / recommend.parse -> 해석 틀 ----------------
INTENT_MAP = {"overview": "개요", "skill": "스킬", "patch": "패치", "stats": "능력치", "build": "템트리",
              "item": "아이템", "role": "추천"}
POS_MAP = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "BOTTOM": "원딜", "UTILITY": "서폿"}
ROLE_MAP = {"Tank": "탱커", "Fighter": "전사", "Mage": "마법사", "Assassin": "암살자", "Marksman": "원딜",
            "Support": "서포터"}


def rule_frame(q, prev=None):
    """지금 서비스의 규칙 해석을 틀로 옮긴다. 규칙은 앞 대화를 보지 않고(prev 무시), 거절하지 않는다(범위는 항상 롤).
    규칙에 없는 칸(방향·개수·난이도·이어받음)은 비워 둔다."""
    import search as S
    import recommend as RC
    P = S.parse(q)
    f = dict(EMPTY, 의도=[], 챔피언=[])
    if P.intent in INTENT_MAP:
        f["의도"] = [INTENT_MAP[P.intent]]
        if P.intent == "patch" and P.why:
            f["의도"] = ["이유"]
    if P.intent == "build":
        mine, enemy, pos = RC.parse(q)
        f["챔피언"] = [{"이름": n, "역할": r} for n, r in ((mine, "나"), (enemy, "상대")) if n]
        f["라인"] = POS_MAP.get(pos)
    elif P.champion:
        f["챔피언"] = [{"이름": P.champion, "역할": "대상"}]
    f["아이템"] = P.item if P.intent == "item" else None
    f["스킬"] = P.slot
    f["역할군"] = ROLE_MAP.get(P.role)
    if P.intent == "patch":
        f["시점"] = "최신" if getattr(P, "patch_auto", False) or not P.patch else P.patch
    return f


# ---------------- 채점 ----------------
def champs_key(f):
    return sorted((c["이름"], c["역할"]) for c in f.get("챔피언") or [])


def score(rows, predict):
    n = len(rows)
    c = Counter()
    per_scope = Counter()
    wrong = []
    for r in rows:
        g, p = r["틀"], predict(r["질문"], r.get("이전") and r["이전"]["틀"])
        ok = {}
        ok["범위"] = p["범위"] == g["범위"]
        gi, pi = g["의도"], p["의도"]
        ok["첫 의도"] = (gi[:1] == pi[:1])
        ok["의도 집합"] = set(gi) == set(pi)
        ok["챔피언·역할"] = champs_key(p) == champs_key(g)
        for k in SCALAR[1:]:
            ok[k] = p.get(k) == g.get(k)
        ok["틀 전체"] = all(ok[k] for k in ["범위", "의도 집합", "챔피언·역할"] + SCALAR[1:])
        # 서비스 관점: 범위가 맞고, 롤이면 첫 의도와 챔피언 역할까지 맞아야 '맞는 답을 고를 수 있는' 해석
        ok["답 가능 해석"] = ok["범위"] and (g["범위"] != "롤" or (ok["첫 의도"] and ok["챔피언·역할"]))
        c.update(k for k, v in ok.items() if v)
        per_scope[(g["범위"], ok["범위"])] += 1
        if not ok["답 가능 해석"]:
            wrong.append((r["질문"], g, p))
    acc = {k: c[k] / n for k in ["답 가능 해석", "범위", "첫 의도", "의도 집합", "챔피언·역할"] + SCALAR[1:] + ["틀 전체"]}
    return acc, per_scope, wrong


def load_gold(path=GOLD):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def brief(f):
    ch = ",".join(f"{c['이름']}:{c['역할']}" for c in f["챔피언"])
    extra = " ".join(f"{k}={f[k]}" for k in SCALAR[1:] if f.get(k) not in (None, False))
    return f"[{f['범위']}] {'+'.join(f['의도']) or '-'} {ch} {extra}".strip()


if __name__ == "__main__":
    rows = load_gold()
    acc, per_scope, wrong = score(rows, rule_frame)
    print(f"정답 세트 {len(rows)}개 — 규칙 기준선 (search.parse)")
    for k, v in acc.items():
        print(f"  {k:10} {v:6.1%}")
    print("  범위별:", {f"{g}{'✓' if o else '✗'}": n for (g, o), n in sorted(per_scope.items())})
    if "--show" in sys.argv:
        for q, g, p in wrong:
            print(f"\n  {q}\n    정답 {brief(g)}\n    규칙 {brief(p)}")
