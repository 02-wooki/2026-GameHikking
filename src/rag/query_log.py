# src/rag/query_log.py
"""질문 로거: 사용자가 입력한 질문을 JSON Lines(logs/queries.jsonl, 한 줄 = 질문 하나)로 저장한다.

목적: 실험 10(MiniGPT 의도 분류)의 평가·학습용 '실제 사람 말투' 질문 모으기.
- label: "question"(제대로 물어본 질문) / "noise"(의미 없는 입력)
  noise_type: mash(막 친 글자) / test(테스트·인사) / too_short / repeat(같은 글자 반복) / no_letters(숫자·기호만)
  ※ 자동 판정이라 틀릴 수 있다. 원문(query)을 그대로 남기므로 나중에 사람이 고칠 수 있다.
- understood: 지금 규칙 기반 해석이 챔피언·아이템·역할군·패치 중 하나라도 알아들었는지
  (question인데 understood=false인 줄 = 지금 서비스가 놓치는 말투 -> 우선 검토 대상)
- check: 답변 자동 점검 {"status": ok / suspect / fail, "reasons": [...]}
  해석 결과와 실제로 고른 근거가 어긋나는지 본다 (챔피언·슬롯·패치 불일치, 조건 완화, 핵심어 미반영,
  템트리 라인 불일치, MiniGPT와 규칙의 선택이 다름 등). 해석부터 틀렸는데 답이 그 해석과 일관되면 못 잡는다.
- 👍/👎: 답변 아래 버튼 -> logs/feedback.jsonl 에 {id, vote} 로 따로 쌓는다 (마지막 투표가 유효)
- 다시 물어봄: 같은 세션에서 2분 안에 비슷한 질문을 다시 하면 앞 답변이 불만족이었을 가능성 -> 요약 때 계산
- intent_gold: 비워 둠. 나중에 사람이 정답 의도를 채우는 칸.
IP·계정 같은 개인정보는 남기지 않는다. 세션 ID는 브라우저 탭마다 새로 만드는 무작위 값.

모으기 결과 요약:
  /usr/local/bin/python3.10 src/rag/query_log.py            # 개수·라벨별 요약 + 놓친 질문 목록
"""
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.environ.get("GH_LOG_DIR", ROOT / "logs"))
LOG_PATH = LOG_DIR / "queries.jsonl"
FEEDBACK_PATH = LOG_DIR / "feedback.jsonl"
KST = timezone(timedelta(hours=9))
_lock = threading.Lock()

TEST_WORDS = {"테스트", "테스트중", "테스트입니다", "test", "testing", "시험", "ㅌㅅㅌ", "ㅌㅅㅌㅌ",
              "안녕", "안녕하세요", "하이", "ㅎㅇ", "hi", "hello", "헬로", "asdf", "qwer", "1234", "123"}
CHAT_JAMO = re.compile(r"([ㄱ-ㅎㅏ-ㅣ])\1+|ㅇㅋ|ㄱㅅ|ㅈㅅ|ㄴㄴ|ㅇㅇ|ㄱㄱ|ㅎㅇ|ㅂㅂ")   # ㅋㅋ, ㅠㅠ 같은 채팅 표현은 정상
KEY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _latin_mash(word):
    w = word.lower()
    if len(w) < 4:
        return False
    if not re.search(r"[aeiouy]", w):
        return True
    return any(w[i:i + 4] in row for row in KEY_ROWS for i in range(len(w) - 3))


def classify(q, known_words=(), understood=False):
    """(label, noise_type) — label은 'question' 또는 'noise'
    understood: 챔피언·아이템 등을 알아들은 입력이면 자음이 섞여도('아리 ㅂㄷ') 막 친 글자로 보지 않는다"""
    t = q.strip()
    compact = re.sub(r"\s+", "", t)
    if len(compact) <= 1:
        return "noise", "too_short"
    if not re.search(r"[가-힣ㄱ-ㅣA-Za-z]", compact):
        return "noise", "no_letters"
    if compact.lower().strip("!?.~") in TEST_WORDS:
        return "noise", "test"
    if (len(set(compact)) == 1 and len(compact) >= 2) or (len(set(compact)) <= 2 and len(compact) >= 4):
        return "noise", "repeat"
    rest = CHAT_JAMO.sub("", compact)
    jamo = len(re.findall(r"[ㄱ-ㅣ]", rest))
    syl = len(re.findall(r"[가-힣]", rest))
    if jamo >= 2 and jamo / max(1, jamo + syl) >= 0.25 and not understood:
        return "noise", "mash"
    words = re.findall(r"[A-Za-z]+", t)
    known = {w.lower() for w in known_words}
    if words and all(_latin_mash(w) and w.lower() not in known for w in words) and syl == 0:
        return "noise", "mash"
    return "question", None


EXPECT_TYPES = {"patch": {"patch_change", "patch_context"}, "skill": {"skill", "profile", "tips"},
                "stats": {"stats"}, "item": {"item"}, "role": {"profile"}}
LANE_WORDS = ("탑", "정글", "미드", "원딜", "바텀", "봇", "서폿", "서포터")


def _lane_of(q):
    import recommend as R
    return next((R.POS_WORDS[w] for w in sorted(R.POS_WORDS, key=len, reverse=True) if w in q), None)


def check(r, rule_best=None):
    """답변 자동 점검 -> (status, reasons). status: ok / suspect / fail"""
    import search as S
    P, hits, best = r.get("parsed"), r.get("hits") or [], r.get("best")
    if P is None:
        return "fail", ["해석 결과 없음"]
    reasons = []
    if P.intent == "free":
        reasons.append("의도를 알아듣지 못함")
    if P.patch_unknown:
        reasons.append(f"없는 패치({P.patch_unknown})")
    names = {n for n in S.NAMES if n in P.query}
    names = {n for n in names if not any(n != m and n in m for m in names)}     # '이'처럼 다른 이름 안에 든 것 제외
    if len(names) >= 2 and P.intent != "build":
        reasons.append(f"챔피언 {len(names)}명이 나왔는데 템트리 질문으로 처리 안 됨")

    if P.intent in ("build", "overview"):
        b = r.get("build")
        if not b:
            return ("fail" if P.intent == "build" else "suspect"), reasons + ["템트리를 만들지 못함"]
        lane = _lane_of(P.query)
        if lane and b.get("position") != lane:
            reasons.append(f"질문 라인({lane})과 템트리 라인({b.get('position')})이 다름")
        return ("suspect" if reasons else "ok"), reasons

    if best is None or best >= len(hits):
        return "fail", reasons + ["근거를 찾지 못함"]
    m = hits[best]["meta"]
    if r.get("relaxed"):
        reasons.append("검색 조건 완화: " + ", ".join(r["relaxed"]))
    if P.champion and m.get("champion") and m["champion"] != P.champion:
        reasons.append(f"챔피언 불일치(질문 {P.champion} / 근거 {m['champion']})")
    if P.slot and m.get("slot") and m["slot"] != P.slot and P.intent in ("patch", "skill"):
        reasons.append(f"슬롯 불일치(질문 {P.slot} / 근거 {m['slot']})")
    if P.patch and m.get("patch") and m["patch"] != P.patch:
        reasons.append(f"패치 불일치(질문 {P.patch} / 근거 {m['patch']})")
    exp = EXPECT_TYPES.get(P.intent)
    if exp and m.get("type") not in exp:
        reasons.append(f"의도({P.intent})와 근거 종류({m.get('type')})가 다름")
    kws = [k for k in P.keywords if not any(k.startswith(w) for w in LANE_WORDS)]
    if kws and P.intent in ("patch", "skill") and not any(k in (r.get("answer") or "") for k in kws):
        reasons.append("질문 핵심어가 답변에 없음: " + ", ".join(kws))
    if rule_best is not None and rule_best != best:
        reasons.append(f"MiniGPT({best + 1}번)와 규칙({rule_best + 1}번)의 선택이 다름")
    return ("suspect" if reasons else "ok"), reasons


def _answer_head(r):
    a = (r.get("answer") or "").strip().splitlines()
    return re.sub(r"[*_]", "", a[0])[:80] if a else ""


def record(q, mode, r, session, turn, source="typed", minigpt=False, rule_best=None, known_words=()):
    """질문 한 개를 한 줄로 추가하고 그 줄의 id를 돌려준다 (👍/👎 연결용).
    로깅 실패가 서비스를 멈추면 안 되므로 예외는 삼키고 None을 돌려준다."""
    try:
        log_id = uuid.uuid4().hex[:12]
        P = r.get("parsed")
        understood = bool(P and (P.champion or P.item or P.role or P.patch or P.intent != "free"))
        label, noise = classify(q, known_words, understood)
        best = r.get("best")
        hits = r.get("hits") or []
        status, reasons = check(r, rule_best) if label == "question" else ("noise", [])
        row = {
            "id": log_id,
            "ts": datetime.now(KST).isoformat(timespec="seconds"),
            "session": session,
            "turn": turn,
            "source": source,                          # typed = 직접 입력, example = 예시 버튼
            "query": q,
            "mode": mode,
            "label": label,
            "noise_type": noise,
            "understood": understood,
            "parsed": None if P is None else {
                "normalized": P.query, "intent": P.intent, "champion": P.champion, "patch": P.patch,
                "slot": P.slot, "item": P.item, "role": P.role, "why": P.why, "keywords": P.keywords},
            "answer": {
                "found": best is not None,
                "type": hits[best]["meta"].get("type") if best is not None and best < len(hits) else None,
                "relaxed": r.get("relaxed") or [],
                "head": _answer_head(r)},
            "check": {"status": status, "reasons": reasons},
            "minigpt": minigpt,
            "intent_gold": None,
        }
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _lock, LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return log_id
    except Exception as ex:                            # noqa: BLE001
        print(f"[query_log] 기록 실패: {ex}")
        return None


def feedback(log_id, session, vote):
    """vote: 1 = 👍, 0 = 👎, None = 취소"""
    try:
        row = {"id": log_id, "session": session, "vote": vote,
               "ts": datetime.now(KST).isoformat(timespec="seconds")}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _lock, FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as ex:                            # noqa: BLE001
        print(f"[query_log] 피드백 기록 실패: {ex}")


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _bigrams(t):
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


def load():
    """질문 로그 + 👍/👎(마지막 투표) + 다시 물어봄 표시를 합쳐서 돌려준다"""
    rows = _jsonl(LOG_PATH)
    votes = {}
    for f in _jsonl(FEEDBACK_PATH):
        votes[f["id"]] = f["vote"]
    for x in rows:
        x["vote"] = votes.get(x.get("id"))
        x["rephrased"] = False
    by_s = {}
    for x in rows:
        if x["source"] == "typed" and x["label"] == "question":
            by_s.setdefault(x["session"], []).append(x)
    for xs in by_s.values():
        xs.sort(key=lambda x: x["ts"])
        for a, b in zip(xs, xs[1:]):
            gap = (datetime.fromisoformat(b["ts"]) - datetime.fromisoformat(a["ts"])).total_seconds()
            pa, pb = a["parsed"] or {}, b["parsed"] or {}
            ba, bb = _bigrams(a["query"]), _bigrams(b["query"])
            similar = len(ba & bb) / len(ba | bb) >= 0.5 or (
                pa.get("champion") and pa.get("champion") == pb.get("champion")
                and pa.get("slot") == pb.get("slot") and pa.get("intent") != pb.get("intent"))
            if gap <= 120 and similar:
                a["rephrased"] = True          # 바로 비슷하게 다시 물음 -> 앞 답변 불만족 가능성
    return rows


if __name__ == "__main__":
    from collections import Counter
    rows = load()
    typed = [x for x in rows if x["source"] == "typed"]
    print(f"{LOG_PATH}\n전체 {len(rows)}줄 (직접 입력 {len(typed)}, 예시 버튼 {len(rows) - len(typed)}) | "
          f"세션 {len({x['session'] for x in rows})}개")
    print("라벨:", dict(Counter(x["label"] if x["label"] == "question" else f"noise/{x['noise_type']}" for x in typed)))
    qs = [x for x in typed if x["label"] == "question"]
    print("의도(규칙 해석):", dict(Counter(x["parsed"]["intent"] for x in qs if x["parsed"])))
    print("자동 점검:", dict(Counter(x["check"]["status"] for x in qs)),
          "| 👍", sum(x["vote"] == 1 for x in qs), "👎", sum(x["vote"] == 0 for x in qs),
          "| 다시 물어봄", sum(x["rephrased"] for x in qs))
    bad = [x for x in qs if x["check"]["status"] != "ok" or x["vote"] == 0 or x["rephrased"]]
    print(f"\n검토할 질문 {len(bad)}개 (자동 점검 suspect/fail, 👎, 다시 물어봄 중 하나라도):")
    for x in bad[-40:]:
        tags = [x["check"]["status"]] + (["👎"] if x["vote"] == 0 else []) + (["재질문"] if x["rephrased"] else [])
        why = "; ".join(x["check"]["reasons"])
        print(f"  - [{'/'.join(tags)}] {x['query']}" + (f"  ({why})" if why else ""))
    print("\nnoise로 분류된 입력 (오분류 확인용, 최근 20개):")
    for x in [x for x in typed if x["label"] == "noise"][-20:]:
        print(f"  - [{x['noise_type']}] {x['query']}")
