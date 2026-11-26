"""실험 11: 선생님 LLM이 만든 질문·해석 틀 데이터 검증 + 학습/검증 세트 나누기

입력: 선생님 LLM 출력 JSONL (여러 파일 가능). 한 줄 = {"이전": {"질문", "틀"} | null, "질문": "...", "틀": {...}}
  - 이름은 실제 이름으로 쓴다 (자리표시 X). 틀 형식은 exp11_frame_spec.md 3장.
검사 (하나라도 걸리면 버리고 이유를 센다):
  1. JSON·필드 형식, 값이 허용 목록 안인지 (의도 11개, 스킬·라인 등)
  2. 범위가 롤이 아니면 의도가 비어 있어야 함 / 롤이면 의도 1~3개
  3. 틀의 챔피언·아이템이 우리 사전에 있는 이름인지, 질문(또는 이전 틀)에서 사전으로 실제로 찾아지는지
     (선생님이 쓴 이름을 우리 사전이 못 찾으면 서비스에서도 못 찾으므로 버림)
  4. 이어받음=true인데 이전 틀이 없으면 버림
  5. 실제 질문 평가 세트(logs/eval/exp11_real_gold.jsonl)와 같은 질문은 버림 (평가 오염 방지), 중복 질문 제거
출력: data/frame11/train.jsonl, data/frame11/val.jsonl (홀드아웃 챔피언이 나오는 문항 = 처음 보는 이름 일반화 확인용)

실행 (프로젝트 루트):  python3.10 src/gpt/experiments/frame11_validate.py data/frame11/raw/*.jsonl
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "gpt"))
sys.path.insert(0, str(ROOT / "src" / "rag"))
sys.path.insert(0, str(ROOT / "src" / "gpt" / "experiments"))

import frame_eval as FE
from frame_model import HEADS, INTENTS, ROLES, Masker, to_example
from lol_tokenizer import load as load_tokenizer

OUT = ROOT / "data" / "frame11"
HOLDOUT_FRAC = 0.12          # 챔피언의 약 12%를 검증 전용으로 (학습 세트에는 그 챔피언이 나오는 문항이 없음)
SEED = 11
FIELDS = ["범위", "의도", "챔피언", "아이템", "스킬", "라인", "역할군", "난이도", "시점", "방향", "개수", "이어받음"]


def norm_q(q):
    return re.sub(r"[\s?!.~]+", "", q).lower()


def check_frame(f, masker):
    """틀 형식 검사 -> 문제 이유 문자열 또는 None. 누락 필드는 기본값으로 채운다(선생님이 null 칸을 생략하는 경우)."""
    if not isinstance(f, dict):
        return "틀이 객체가 아님"
    for k in FIELDS:
        f.setdefault(k, FE.EMPTY[k] if k not in ("의도", "챔피언") else [])
    extra = set(f) - set(FIELDS)
    if extra:
        return f"모르는 필드 {sorted(extra)[0]}"
    for k, vals in HEADS.items():
        v = f[k]
        if k == "시점" and isinstance(v, str) and masker.S.PATCH_RE.fullmatch(v.strip()):
            continue                                   # 실제 패치 번호
        if v not in vals:
            return f"{k} 값 {v!r}"
    if not isinstance(f["의도"], list) or any(i not in INTENTS for i in f["의도"]):
        return f"의도 값 {f['의도']!r}"
    if len(set(f["의도"])) != len(f["의도"]):
        return "의도 중복"
    if f["범위"] == "롤" and not 1 <= len(f["의도"]) <= 3:
        return "롤인데 의도 개수가 1~3이 아님"
    if f["범위"] != "롤" and f["의도"]:
        return "롤이 아닌데 의도가 있음"
    if not isinstance(f["챔피언"], list):
        return "챔피언이 목록이 아님"
    for c in f["챔피언"]:
        if not isinstance(c, dict) or c.get("역할") not in ROLES[1:] or c.get("이름") not in masker.S.CHAMPS:
            return f"챔피언 항목 {c!r}"
    if len({c["이름"] for c in f["챔피언"]}) != len(f["챔피언"]):
        return "같은 챔피언이 두 번"
    if f["아이템"] is not None and f["아이템"] not in set(masker.S.ITEM_NAMES):
        return f"모르는 아이템 {f['아이템']!r}"
    return None


def main(paths):
    tok = load_tokenizer(str(ROOT / "tokenizer_out"))
    masker = Masker()
    gold_q = {norm_q(r["질문"]) for r in FE.load_gold()}
    why, seen, kept = Counter(), set(), []
    n_in = 0
    for p in paths:
        for i, line in enumerate(open(p, encoding="utf-8")):
            if not line.strip():
                continue
            n_in += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                why["JSON 깨짐"] += 1
                continue
            if not isinstance(r, dict) or not isinstance(r.get("질문"), str) or not r["질문"].strip():
                why["질문 없음"] += 1
                continue
            prev = r.get("이전")
            if prev is not None:
                if not isinstance(prev, dict) or check_frame(prev.get("틀"), masker):
                    why["이전 틀 형식"] += 1
                    continue
            bad = check_frame(r.get("틀"), masker)
            if bad:
                why["틀 형식: " + bad.split(" ")[0]] += 1
                continue
            if r["틀"]["이어받음"] and not prev:
                why["이어받음인데 이전 없음"] += 1
                continue
            k = norm_q(r["질문"])
            if k in gold_q:
                why["실제 평가 질문과 같음"] += 1
                continue
            key = (k, json.dumps(prev and prev.get("틀"), ensure_ascii=False, sort_keys=True))
            if key in seen:
                why["중복"] += 1
                continue
            if to_example(masker, tok, r, 192) is None:
                why["틀의 이름을 사전이 질문에서 못 찾음"] += 1
                continue
            seen.add(key)
            kept.append(r)

    champs = sorted({c["이름"] for r in kept for c in r["틀"]["챔피언"]})
    rng = random.Random(SEED)
    hold = set(rng.sample(champs, int(len(champs) * HOLDOUT_FRAC))) if champs else set()

    def names(r):
        fs = [r["틀"]] + ([r["이전"]["틀"]] if r.get("이전") else [])
        return {c["이름"] for f in fs for c in f["챔피언"]}

    train = [r for r in kept if not names(r) & hold]
    val = [r for r in kept if names(r) & hold]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"입력 {n_in}줄 -> 통과 {len(kept)} ({len(kept) / max(1, n_in):.0%}) | 학습 {len(train)} / 검증 {len(val)} "
          f"(홀드아웃 챔피언 {len(hold)}명)")
    if why:
        print("버린 이유:", dict(why.most_common()))
    if kept:
        sc = Counter(r["틀"]["범위"] for r in kept)
        print("범위:", {k: f"{v / len(kept):.0%}" for k, v in sc.items()}, "(목표: 사이트밖 15%, 데이터밖 10%)")
        print("첫 의도:", dict(Counter(r["틀"]["의도"][0] for r in kept if r["틀"]["의도"]).most_common()))
        print(f"의도 2개 이상 {sum(len(r['틀']['의도']) > 1 for r in kept) / len(kept):.0%} (목표 20%) | "
              f"이전 있음 {sum(bool(r.get('이전')) for r in kept) / len(kept):.0%} (목표 20%) | "
              f"이어받음 {sum(r['틀']['이어받음'] for r in kept) / len(kept):.0%}")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(str(p) for p in (OUT / "raw").glob("*.jsonl"))
    if not args:
        sys.exit("입력 파일이 없습니다: data/frame11/raw/*.jsonl 또는 경로를 인자로")
    main(args)
