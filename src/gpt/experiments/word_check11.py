"""실험 11: 선생님(EXAONE) 질문에서 '지어낸 단어'(예: '쁠뤽')가 든 줄을 자동으로 버린다 — 사람·Claude가 한 줄씩 안 봐도 되게.

방법: 질문의 한글 낱말을 '아는 조각'으로 끝까지 쪼갤 수 있는지 본다 (띄어쓰기 안 한 '자야상향이유'도 자야+상향+이유로 통과).
  아는 조각 = 우리 말뭉치(패치노트·챔피언·아이템 jsonl) 낱말의 2~4글자 부분 문자열
            + 선생님 데이터 전체(--pool)에서 MIN_COUNT번 이상 나온 낱말의 부분 문자열 (자주 쓰는 신조어·줄임말은 살림)
            + 게이머 말투 목록, 한 글자 조사·어미
  쪼갤 수 없으면(모르는 음절 조합이 남으면) 버린다. 사이트밖(잡담)은 롤 말뭉치에 없는 일상어가 많아 검사하지 않음.
  ㅋㅋ·ㄱㄱ 같은 자음만은 검사 안 함.

실행 (프로젝트 루트):
  py -3.10 src/gpt/experiments/word_check11.py data/frame11/raw/exaone_run3.jsonl --pool "data/frame11/raw/exaone_run*.jsonl"
  -> exaone_run3.words.jsonl (통과 줄), 버린 낱말 상위 목록 출력 (--show: 버린 줄 전부)
"""
import argparse
import glob
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = [ROOT / "data" / "processed" / n for n in ("patch_notes.jsonl", "champions.jsonl", "items.jsonl")]
MIN_COUNT = 3
HANGUL = re.compile(r"[가-힣]+")
SLANG = """어케 어캐 어떰 머임 뭐임 뭐냐 머냐 머야 뭐야 뭐감 쿨탐 딜교 한타 왕귀 갱킹 라인전 맞라인 카운터 너프먹 상향됐 하향됐
받았냐 좋냐 좋음 구림 구려 꿀챔 고인물 뉴비 롤린이 초보 입문 꿀팁 알려줘 정리 요즘 최근 이번 할만 해볼 괜찮 진짜 개좋 쎄냐
세냐 떡상 떡락 픽률 승률 몇초 얼마 총합 시작템 첫템 신발 서폿 그랬 그럼 그거 이거 저거 근데 어때 어떻 어떤 무슨 왜케 왤케
궁금 같이 동시 사용 가능 확인 계산 순서 범위 효과 작동 변천 변화 사항 차이 비교 설명 추천 해줘 주세요 알려 있어 없어 할까""".split()
ONE = set("은는이가을를의에로도만좀요야냐지해함임다고면서랑과와게네죠줘거건걸뭐왜몇등임음냐니나까데듯한할된될던든런렇께엔앤감됨씀셈여돼되쓰")


def subs(w, lo=2, hi=4):
    return {w[i:j] for i in range(len(w)) for j in range(i + lo, min(len(w), i + hi) + 1)}


def build_known(pool_words):
    known = set()
    for p in CORPUS:
        if p.exists():
            for w in set(HANGUL.findall(p.read_text(encoding="utf-8"))):
                known |= subs(w)
                if len(w) == 1:
                    known.add(w)                        # 한 글자 낱말 (궁·템·수·때 등)
    for w in SLANG:
        known |= subs(w)
    for w, c in pool_words.items():
        if c >= MIN_COUNT:
            known |= subs(w)
            if len(w) == 1:
                known.add(w)
    return known


def make_ok(known):
    @lru_cache(maxsize=None)
    def ok(w):
        """w를 아는 조각(2~4글자) + 한 글자 조사·어미로 끝까지 쪼갤 수 있나"""
        if not w:
            return True
        if (w[0] in ONE or w[0] in known) and ok(w[1:]):
            return True
        return any(w[:k] in known and ok(w[k:]) for k in range(min(4, len(w)), 1, -1))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--pool", nargs="*", default=[], help="빈도를 셀 선생님 데이터 (glob 가능)")
    ap.add_argument("--show", action="store_true", help="버린 줄 전부 출력")
    a = ap.parse_args()
    files = sorted({p for g in a.pool for p in glob.glob(g)} | {a.file})
    pool = Counter()
    for p in files:
        if p.endswith(".words.jsonl"):
            continue
        for l in open(p, encoding="utf-8"):
            pool.update(HANGUL.findall(json.loads(l)["질문"]))
    ok = make_ok(build_known(pool))
    rows = [json.loads(l) for l in open(a.file, encoding="utf-8")]
    out = Path(a.file).with_suffix(".words.jsonl")
    bad_words, dropped = Counter(), []
    with open(out, "w", encoding="utf-8") as fo:
        for r in rows:
            bad = [] if r["틀"]["범위"] == "사이트밖" else [w for w in HANGUL.findall(r["질문"]) if not ok(w)]
            if bad:
                bad_words.update(bad)
                dropped.append((r["질문"], bad))
            else:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(f"{a.file}: {n}줄 -> 통과 {n - len(dropped)} / 버림 {len(dropped)} ({len(dropped) / max(n, 1):.1%})  -> {out}")
    print("버린 낱말 상위:", ", ".join(f"{w}({c})" for w, c in bad_words.most_common(50)))
    if a.show:
        for q, b in dropped:
            print(f"  {q}   [{', '.join(b)}]")


if __name__ == "__main__":
    main()
