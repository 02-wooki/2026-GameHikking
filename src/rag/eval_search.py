# src/rag/eval_search.py
"""검색 v1(기존 데모 방식) vs v2 정량 비교
- v1: 'gamehikking' 컬렉션, 벡터 상위 5개 + 키워드 재정렬 (현재 app.py와 같은 로직), 챔피언 단위 큰 청크
- v2: 'gamehikking_v2' 컬렉션, search.answer() (질문 해석 + 필터 + 규칙 기반 줄 선택), 줄·스킬 단위 청크

채점 (최종 선택된 근거 1개 기준 = top1):
  champion 일치, 문서 종류 일치, patch 일치(지정 시), slot 일치(v2만, 지정 시), field 문자열 포함(지정 시)
  ※ v1은 청크가 챔피언 전체라 slot은 채점하지 않음 (v1에 유리한 조건)
  ※ v1 DB는 26-16~26-20 패치만 있으므로, 그 범위 문항만 따로 비교한 수치도 출력

실행 (프로젝트 루트에서, build_index.py 실행 후):
  /usr/local/bin/python3.10 src/rag/eval_search.py           # 개발용 세트 (26문항)
  /usr/local/bin/python3.10 src/rag/eval_search.py --user    # 사용자 말투 세트 (10문항)
  /usr/local/bin/python3.10 src/rag/eval_search.py --item    # 아이템 세트 (6문항)
"""
import json
from pathlib import Path

import search as S

V1_PATCHES = {"26-16", "26-17", "26-18", "26-19", "26-20"}
LATEST = "latest"

# (질문, 기대값) — kind: patch / context / skill / stats / role
CASES = [
    ("아트록스 26.19 패치 W 재사용 대기시간 어떻게 바뀌었어?", dict(kind="patch", champion="아트록스", patch="26-19", slot="W", field="재사용 대기시간")),
    ("벨베스 26-16 궁극기 본모습 총공격 속도 변경", dict(kind="patch", champion="벨베스", patch="26-16", slot="R", field="본모습 총공격 속도")),
    ("니코 26.20 E 강화 속박 지속시간", dict(kind="patch", champion="니코", patch="26-20", slot="E", field="강화 속박 지속시간")),
    ("마스터이 26.18 궁 이동 속도 바뀜?", dict(kind="patch", champion="마스터 이", patch="26-18", slot="R", field="이동 속도")),
    ("럼블 26-19 패시브 과열 상태 공격 속도", dict(kind="patch", champion="럼블", patch="26-19", slot="패시브", field="과열 상태 공격 속도")),
    ("드레이븐 26.19 기본 공격력 상향됐어?", dict(kind="patch", champion="드레이븐", patch="26-19", field="공격력")),
    ("다이애나 26-20 W 보호막 추가 체력 계수", dict(kind="patch", champion="다이애나", patch="26-20", slot="W", field="보호막 추가 체력 계수")),
    ("신드라 26-18 패시브 분노의 조각 마나", dict(kind="patch", champion="신드라", patch="26-18", slot="패시브", field="분노의 조각 마나")),
    ("잭스 13.1 패치 Q 물리 피해량", dict(kind="patch", champion="잭스", patch="13-01", slot="Q", field="물리 피해량")),
    ("흐웨이 14.1 패치 기본 방어력", dict(kind="patch", champion="흐웨이", patch="14-01", field="기본 방어력")),
    ("나서스 26-19 패치에서 뭐 바뀌었어?", dict(kind="patch", champion="나서스", patch="26-19")),
    ("카시오페아 26.18 패치 변경점", dict(kind="patch", champion="카시오페아", patch="26-18")),
    ("베인 최근 패치 변경사항 알려줘", dict(kind="patch", champion="베인", patch=LATEST)),
    ("뽀삐 너프 내용", dict(kind="patch", champion="뽀삐", patch=LATEST)),
    ("잭스 13.1 패치 왜 바뀐 거야?", dict(kind="context", champion="잭스", patch="13-01")),
    ("아리 궁극기 뭐야?", dict(kind="skill", champion="아리", slot="R")),
    ("나서스 Q 스킬 알려줘", dict(kind="skill", champion="나서스", slot="Q")),
    ("요네 W 뭐야", dict(kind="skill", champion="요네", slot="W")),
    ("아펠리오스 E 설명해줘", dict(kind="skill", champion="아펠리오스", slot="E")),
    ("레나타 R 뭐야?", dict(kind="skill", champion="레나타 글라스크", slot="R")),
    ("럭스 궁 알려줘", dict(kind="skill", champion="럭스", slot="R")),
    ("아트록스 패시브", dict(kind="skill", champion="아트록스", slot="패시브")),
    ("아트록스 기본 체력 얼마야?", dict(kind="stats", champion="아트록스")),
    ("나서스 공격력 몇이야", dict(kind="stats", champion="나서스")),
    ("탱커 챔피언 추천해줘", dict(kind="role", role="Tank")),
    ("원딜 챔피언 뭐 있어?", dict(kind="role", role="Marksman")),
]

# 사용자 말투 세트: 규칙에 맞추지 않고 실제 사용자가 칠 법한 표현 (줄임말, 스킬 이름, 영어, 소문자 등)
USER_CASES = [
    ("아트록스 지옥사슬 쿨 줄었어?", dict(kind="patch", champion="아트록스", patch=LATEST, slot="W", field="재사용 대기시간")),
    ("마이 궁 이속 얼마나 올랐음?", dict(kind="patch", champion="마스터 이", patch=LATEST, slot="R", field="이동 속도")),
    ("13.1 잭스 반격 피해량 어떻게 바뀜", dict(kind="patch", champion="잭스", patch="13-01", slot="E", field="피해량")),
    ("니코 칭칭올가미 바뀐거 있어?", dict(kind="patch", champion="니코", patch=LATEST, slot="E")),
    ("다이애나 e 공속 26.20", dict(kind="patch", champion="다이애나", patch="26-20", slot="E", field="공격 속도")),
    ("흐웨이 방어력 버프됨?", dict(kind="patch", champion="흐웨이", patch=LATEST, field="방어력")),
    ("럼블 패시브 설명 좀", dict(kind="skill", champion="럼블", slot="패시브")),
    ("Aatrox Q 뭐임", dict(kind="skill", champion="아트록스", slot="Q")),
    ("아리 체력 몇임", dict(kind="stats", champion="아리")),
    ("서포터 챔프 목록", dict(kind="role", role="Support")),
]

# 아이템 세트 (W9)
ITEM_CASES = [
    ("삼위일체 가격 얼마야?", dict(kind="item", item="삼위일체")),
    ("몰왕 조합법 알려줘", dict(kind="item", item="몰락한 왕의 검")),
    ("가시 갑옷 효과", dict(kind="item", item="가시 갑옷")),
    ("존야 얼마임", dict(kind="item", item="존야의 모래시계")),
    ("라바돈의 죽음모자 재료", dict(kind="item", item="라바돈의 죽음모자")),
    ("무한의 대검 능력치 알려줘", dict(kind="item", item="무한의 대검")),
]

V2_TYPE = {"item": "item", "patch": "patch_change", "context": "patch_context", "skill": "skill", "stats": "stats", "role": "profile"}


def resolve(e):
    """'latest'를 실제 패치로 바꾸고, 데이터에 정답이 존재하는지 확인"""
    e = dict(e)
    if e["kind"] == "item":
        return (e, None) if e["item"] in set(S.ITEM_NAMES) else (None, f"아이템 '{e['item']}' 없음")
    c = e.get("champion")
    if c and c not in S.CHAMPS:
        return None, f"챔피언 '{c}' 없음"
    if e.get("patch") == LATEST:
        if c not in S.BY_CHAMP:
            return None, f"{c} 패치 기록 없음"
        # 슬롯·필드 조건이 있으면 '그 항목이 마지막으로 바뀐 패치'가 정답
        match = [r for r in S.BY_CHAMP[c] if any((not e.get("slot") or x["slot"] == e["slot"])
                                                  and (not e.get("field") or e["field"] in x["text"])
                                                  for x in r["changes"])]
        if not match:
            return None, "조건에 맞는 변경 기록 없음"
        e["patch"] = match[0]["patch"]
    if e["kind"] in ("patch", "context"):
        rec = S.RECORDS.get(f"{e['patch']}_{c}")
        if not rec:
            return None, f"{e['patch']}_{c} 레코드 없음"
        if e["kind"] == "context" and not (rec["summary"] or rec["context"]):
            return None, "개발자 코멘트 없음"
        lines = [x for x in rec["changes"] if (not e.get("slot") or x["slot"] == e["slot"])
                 and (not e.get("field") or e["field"] in x["text"])]
        if not lines:
            return None, "조건에 맞는 변경 줄 없음"
    return e, None


def ok_v2(e, h):
    m = h["meta"]
    checks = {"종류": m.get("type") == V2_TYPE[e["kind"]]}
    if e["kind"] == "role":
        checks["역할"] = m.get(f"role_{e['role']}") is True
    elif e["kind"] == "item":
        checks["아이템"] = m.get("name") == e["item"]
    else:
        checks["챔피언"] = m.get("champion") == e["champion"]
    if e.get("patch"):
        checks["패치"] = m.get("patch") == e["patch"]
    if e.get("slot"):
        checks["슬롯"] = m.get("slot") == e["slot"]
    if e.get("field"):
        checks["필드"] = e["field"] in h["text"]
    return all(checks.values()), [k for k, v in checks.items() if not v]


# ---------- v1 (기존 app.py 로직 그대로) ----------
def v1_collection():
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=S.MODEL)
    try:
        return chromadb.PersistentClient(path=str(S.DB_PATH)).get_collection("gamehikking", embedding_function=ef)
    except Exception:
        return None


def v1_answer(col, q):
    res = col.query(query_texts=[q], n_results=5)
    chunks, metas = res["documents"][0], res["metadatas"][0]
    kws = [w for w in q.split() if w not in ["정보", "알려줘", "스킬", "어때", "뭐야", "소개해줘"]] or q.split()
    for cond in (all, any):
        for c, m in zip(chunks, metas):
            if cond(kw in c for kw in kws):
                return c, m
    return (chunks[0], metas[0]) if chunks else (None, None)


def ok_v1(e, text, m):
    if text is None:
        return False, ["결과 없음"]
    if e["kind"] == "item":
        return (m.get("type") == "item" and e["item"] in (m.get("name") or "")), ["아이템"]
    want = "patch_note" if e["kind"] in ("patch", "context") else "base_info"
    checks = {"종류": m.get("type") == want}
    if e["kind"] == "role":
        checks["역할"] = e["role"] in text
    else:
        checks["챔피언"] = m.get("champion") == e["champion"]
    if e.get("patch"):
        checks["패치"] = m.get("patch") == e["patch"]
    if e.get("field"):
        checks["필드"] = e["field"] in text
    return all(checks.values()), [k for k, v in checks.items() if not v]


def main():
    import sys
    global CASES
    if "--user" in sys.argv:
        CASES = USER_CASES
        print("※ 사용자 말투 세트 (10문항)")
    elif "--item" in sys.argv:
        CASES = ITEM_CASES
        print("※ 아이템 세트 (6문항)")
    col1 = v1_collection()
    if col1 is None:
        print("※ v1 컬렉션 'gamehikking'이 없어 v1 비교는 건너뜁니다.")
    rows, skipped = [], []
    for q, e0 in CASES:
        e, why = resolve(e0)
        if e is None:
            skipped.append((q, why))
            continue
        r = S.answer(q)
        v2 = ok_v2(e, r["hits"][r["best"]]) if r["best"] is not None else (False, ["결과 없음"])
        v2_any = any(ok_v2(e, h)[0] for h in r["hits"][:5])
        v1 = ok_v1(e, *v1_answer(col1, q)) if col1 else (None, [])
        in_v1 = e.get("patch") in V1_PATCHES if e["kind"] in ("patch", "context") else True
        rows.append(dict(q=q, kind=e["kind"], v2=v2[0], v2_miss=v2[1], v2_top5=v2_any,
                         v1=v1[0], v1_miss=v1[1], in_v1=in_v1, P=r["parsed"]))

    print(f"\n{'질문':42s} | 종류    | v1 | v2 | v2 실패 항목")
    print("-" * 95)
    for x in rows:
        v1s = "-" if x["v1"] is None else ("O" if x["v1"] else "X")
        print(f"{x['q'][:40]:42s} | {x['kind']:7s} | {v1s:2s} | {'O' if x['v2'] else 'X':2s} | {', '.join(x['v2_miss'])}")

    n = len(rows)
    acc = lambda key, rs: f"{sum(r[key] for r in rs)}/{len(rs)} ({sum(r[key] for r in rs) / max(1, len(rs)) * 100:.1f}%)"
    print(f"\n===== 결과 (유효 문항 {n}개, 제외 {len(skipped)}개) =====")
    print(f"v2 top1 정확도 : {acc('v2', rows)}")
    print(f"v2 top5 포함   : {acc('v2_top5', rows)}")
    if col1:
        print(f"v1 top1 정확도 : {acc('v1', rows)}")
        sub = [r for r in rows if r["in_v1"]]
        print(f"[v1 데이터 범위(26.16~26.20 + 챔피언 정보) 문항만] v1 {acc('v1', sub)} | v2 {acc('v2', sub)}")
    print("\n종류별 v2:", {k: acc("v2", [r for r in rows if r["kind"] == k]) for k in V2_TYPE})
    if skipped:
        print("\n제외된 문항 (데이터에 정답이 없음 -> 문항 수정 필요):")
        for q, why in skipped:
            print(f"  {q}  <- {why}")
    fails = [r for r in rows if not r["v2"]]
    if fails:
        print("\nv2 실패 문항의 질문 해석:")
        for r in fails:
            P = r["P"]
            print(f"  {r['q']}\n    -> 의도 {P.intent} | 챔피언 {P.champion} | 패치 {P.patch} | 슬롯 {P.slot} | 키워드 {P.keywords}")

    out = Path(S.ROOT) / "checkpoints" / ("search_eval_user.json" if CASES is USER_CASES else "search_eval_item.json" if CASES is ITEM_CASES else "search_eval.json")
    out.write_text(json.dumps([{k: v for k, v in r.items() if k != "P"} for r in rows], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()