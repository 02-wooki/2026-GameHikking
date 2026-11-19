# src/rag/search.py
"""검색 v2: 질문 해석 -> 메타데이터 필터 + 벡터 검색 -> 근거 줄 선택 -> 템플릿 응답
app.py, eval_search.py에서 import 해서 사용한다.

  parse(q)            : 챔피언·패치·스킬 슬롯·의도(patch/skill/stats/role/free) 추출
  search(q, mode)     : 필터 검색 (결과가 없으면 조건을 하나씩 완화)
  select_line(P, hits): 근거 선택. 기본은 규칙 기반. set_selector()로 MiniGPT 리랭커 등으로 교체 가능
  answer(q, mode)     : 최종 응답(마크다운) + 근거 + 출처

데이터 기준: 패치노트 = 공식 패치노트 크롤링, 챔피언 정보 = Data Dragon (champions.jsonl의 version)
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
DB_PATH = ROOT / "chroma_db"
COLLECTION = "gamehikking_v2"
MODEL = "distiluse-base-multilingual-cased-v1"

SLOT_LABEL = {"패시브": "기본 지속 효과", "Q": "Q", "W": "W", "E": "E", "R": "R"}
ROLE_WORDS = {"탱커": "Tank", "전사": "Fighter", "브루저": "Fighter", "마법사": "Mage", "메이지": "Mage",
              "암살자": "Assassin", "어쌔신": "Assassin", "원거리 딜러": "Marksman", "원딜": "Marksman",
              "서포터": "Support", "서폿": "Support"}
ROLE_KO = {"Tank": "탱커", "Fighter": "전사", "Mage": "마법사", "Assassin": "암살자",
           "Marksman": "원거리 딜러", "Support": "서포터"}
STAT_WORDS = {"체력": "hp", "마나": "mp", "공격력": "attackdamage", "공격 속도": "attackspeed", "공속": "attackspeed",
              "방어력": "armor", "마법 저항력": "spellblock", "마저": "spellblock", "이동 속도": "movespeed",
              "이동속도": "movespeed", "이속": "movespeed", "사거리": "attackrange", "능력치": None, "스탯": None}
STAT_NAME = {"hp": "체력", "mp": "마나", "attackdamage": "공격력", "attackspeed": "공격 속도", "armor": "방어력",
             "spellblock": "마법 저항력", "movespeed": "이동 속도", "attackrange": "사거리"}
PATCH_WORDS = ("패치", "너프", "버프", "상향", "하향", "변경", "바뀌", "바꼈", "바뀐", "바뀜", "달라",
               "줄었", "줄어", "올랐", "올라", "늘었", "늘어", "낮아", "높아", "증가", "감소", "조정")
WHY_WORDS = ("왜", "이유", "의도")
ITEM_WORDS = ("아이템", "템", "가격", "골드", "조합", "재료", "얼마", "효과", "능력치", "스탯")
BUILD_WORDS = ("템트리", "빌드", "아이템 추천", "템 추천", "뭐 사", "뭘 사", "뭐사", "코어템", "아이템 뭐", "템 뭐")
ITEM_ALIASES = {"몰왕": "몰락한 왕의 검", "가갑": "가시 갑옷", "존야": "존야의 모래시계", "무대": "무한의 대검",
                "라바돈": "라바돈의 죽음모자", "삼위": "삼위일체", "구인수": "구인수의 격노검", "란두인": "란두인의 예언",
                "워모그": "워모그의 갑옷", "스태틱": "스태틱의 단검", "드락사르": "드락사르의 황혼검"}
ALIASES = {"누누": "누누와 윌럼프", "레나타": "레나타 글라스크", "오솔": "아우렐리온 솔", "트페": "트위스티드 페이트",
           "문도": "문도 박사", "미포": "미스 포츈", "블츠": "블리츠크랭크", "블리츠": "블리츠크랭크",
           "자르반": "자르반 4세", "리신": "리 신", "마이": "마스터 이", "이렐": "이렐리아", "케틀": "케이틀린",
           "트타": "트리스타나", "카타": "카타리나", "말파": "말파이트", "모데": "모데카이저", "르블": "르블랑",
           "아우솔": "아우렐리온 솔", "애니비": "애니비아"}
# 줄임말 -> 패치노트·데이터에 실제로 쓰이는 표현 (키워드 매칭용)
ABBR = {"쿨": "재사용 대기시간", "쿨타임": "재사용 대기시간", "쿨다운": "재사용 대기시간", "쿨감": "재사용 대기시간",
        "이속": "이동 속도", "공속": "공격 속도", "마저": "마법 저항력", "딜": "피해량", "데미지": "피해량",
        "뎀지": "피해량", "체젠": "체력 재생", "마젠": "마나 재생", "사정거리": "사거리", "보막": "보호막"}
STOP_PREFIX = ("패치", "변경", "바뀌", "바꼈", "바뀐", "바뀜", "알려", "설명", "뭐", "무엇", "어떻게", "어때", "얼마", "몇",
               "줄었", "줄어", "올랐", "올라", "늘었", "늘어", "낮아", "높아", "조정", "목록", "챔프",
               "상향", "하향", "너프", "버프", "최근", "이번", "내용", "정보", "스킬", "궁극기", "패시브", "기본 지속",
               "추천", "챔피언", "있어", "왜", "이유", "의도")
JOSA = re.compile(r"(은|는|이|가|을|를|의|도|만|에서|으로|로|에)$")
PATCH_RE = re.compile(r"(?<![\d.])(\d{2})\s*[.\-]\s*(?:([sS]\d)\s*[.\-]\s*(\d)|(\d{1,2}))(?![\d])")


# ---------------- 데이터 ----------------
def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


CHAMPS = {c["name"]: c for c in _load_jsonl(PROC / "champions.jsonl")}
NAMES = sorted(CHAMPS, key=len, reverse=True)
NAMES_NOSPACE = sorted(((n.replace(" ", ""), n) for n in NAMES), key=lambda x: -len(x[0]))
ALIASES = dict(sorted(((k, v) for k, v in ALIASES.items() if v in CHAMPS), key=lambda x: -len(x[0])))
EN_NAMES = {c["id"].lower(): c["name"] for c in CHAMPS.values()}            # Aatrox -> 아트록스
RECORDS = {r["id"]: r for r in _load_jsonl(PROC / "patch_notes.jsonl")}
ALL_PATCHES = {r["patch"] for r in RECORDS.values()}
BY_CHAMP = {}
for _r in RECORDS.values():
    BY_CHAMP.setdefault(_r["champion"], []).append(_r)
for _v in BY_CHAMP.values():
    _v.sort(key=lambda r: -r["patch_order"])                   # 최신 패치가 앞
DD_VERSION = next(iter(CHAMPS.values()))["version"]

_items_path = PROC / "items.jsonl"
ITEMS = _load_jsonl(_items_path) if _items_path.exists() else []
ITEM_BY_ID = {it["id"]: it for it in ITEMS}
ITEM_NAMES = sorted({it["name"] for it in ITEMS if len(it["name"]) >= 2}, key=len, reverse=True)
ITEM_ALIASES = dict(sorted(((k, v) for k, v in ITEM_ALIASES.items() if v in set(ITEM_NAMES)), key=lambda x: -len(x[0])))

# 스킬 이름 -> 슬롯 (Data Dragon 현재 이름 + 패치노트에 나온 이름)
SKILL_SLOTS = {}
def _add_skill(champ, name, slot):
    name = (name or "").split(" (")[0].strip()
    if len(name) >= 2 and slot in SLOT_LABEL:
        SKILL_SLOTS.setdefault(champ, {})[name] = slot
for _c in CHAMPS.values():
    for _s in _c["skills"]:
        _add_skill(_c["name"], _s["name"], _s["slot"])
for _r in RECORDS.values():
    for _ch in _r["changes"]:
        _add_skill(_r["champion"], _ch["skill"], _ch["slot"])
_owners = {}
for _c, _m in SKILL_SLOTS.items():
    for _n, _sl in _m.items():
        _owners.setdefault(_n, set()).add((_c, _sl))
UNIQUE_SKILL = {n: next(iter(o)) for n, o in _owners.items() if len(o) == 1 and len(n) >= 3}   # 스킬명만으로 챔피언 추정


@lru_cache(maxsize=1)
def collection():
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)
    return chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION, embedding_function=ef)


def patch_label(p):
    """'13-01' -> '13.1', '25-S1-1' -> '25.S1.1' (공식 표기)"""
    return ".".join(str(int(x)) if x.isdigit() else x for x in p.split("-"))


# ---------------- 질문 해석 ----------------
@dataclass
class Parsed:
    query: str
    champion: str = None
    patch: str = None
    patch_auto: bool = False          # 패치를 안 적어서 '최신 변경 패치'로 채운 경우
    patch_unknown: str = None         # 적었지만 데이터에 없는 패치
    slot: str = None
    intent: str = "free"              # patch / skill / stats / role / free
    why: bool = False
    role: str = None
    stat: str = None
    item: str = None
    keywords: list = field(default_factory=list)


def find_champion(q):
    for n in NAMES:
        if n in q:
            return n
    qs = q.replace(" ", "")
    for ns, n in NAMES_NOSPACE:
        if ns in qs:
            return n
    for a, n in ALIASES.items():
        if a in q:
            return n
    for w in re.findall(r"[A-Za-z]{3,}", q):
        if w.lower() in EN_NAMES:
            return EN_NAMES[w.lower()]
    return None


def find_item(t):
    for n in ITEM_NAMES:
        if n in t:
            return n
    ts = t.replace(" ", "")
    for n in ITEM_NAMES:
        if len(n) >= 4 and n.replace(" ", "") in ts:
            return n
    for a, n in ITEM_ALIASES.items():
        if a in t:
            return n
    return None


def find_slot(t):
    if "궁극기" in t or re.search(r"궁(?!금)", t):
        return "R"
    if "패시브" in t or "기본 지속 효과" in t:
        return "패시브"
    m = re.search(r"(?<![A-Za-z])([QWERqwer])(?![A-Za-z])", t)
    return m.group(1).upper() if m else None


def keywords(t):
    out = []
    for w in re.split(r"[\s,?!.~]+", t):
        b = JOSA.sub("", w) or w
        if w in ABBR or b in ABBR:
            out.append(ABBR.get(w) or ABBR[b])
            continue
        w = JOSA.sub("", w) if len(w) > 2 else w
        if len(w) >= 2 and not w.startswith(STOP_PREFIX):
            out.append(w)
    return out


def parse(q):
    P = Parsed(query=q)
    P.champion = find_champion(q)
    skill_slot = None
    if not P.champion:                                         # 챔피언 없이 고유 스킬명만 적은 경우
        hit = next((n for n in sorted(UNIQUE_SKILL, key=len, reverse=True) if n in q), None)
        if hit:
            P.champion, skill_slot = UNIQUE_SKILL[hit]
    rest = q
    if P.champion:
        en = next(c["id"] for c in CHAMPS.values() if c["name"] == P.champion)
        for name in [P.champion, P.champion.replace(" ", "")] + [a for a, n in ALIASES.items() if n == P.champion]:
            rest = rest.replace(name, " ")
        rest = re.sub(re.escape(en), " ", rest, flags=re.I)
    m = PATCH_RE.search(rest)
    if m:
        p = f"{m[1]}-{m[2].upper()}-{m[3]}" if m[2] else f"{m[1]}-{int(m[4]):02d}"
        if p in ALL_PATCHES:
            P.patch = p
        else:
            P.patch_unknown = patch_label(p)
        rest = rest.replace(m[0], " ")
    P.slot = find_slot(rest)
    if not P.slot and P.champion:                              # 스킬 이름으로 슬롯 찾기 (예: 지옥사슬 -> W)
        names = SKILL_SLOTS.get(P.champion, {})
        hit = next((n for n in sorted(names, key=len, reverse=True) if n in rest), None)
        P.slot = names[hit] if hit else skill_slot
    P.why = any(w in q for w in WHY_WORDS)
    if not P.champion:
        P.role = next((ROLE_WORDS[w] for w in sorted(ROLE_WORDS, key=len, reverse=True) if w in q), None)
    stat_word = next((w for w in sorted(STAT_WORDS, key=len, reverse=True) if w in q), None)
    P.item = find_item(rest)
    if any(w in q for w in BUILD_WORDS):
        P.intent = "build"
    elif P.item and (not P.champion or any(w in q for w in ITEM_WORDS)):
        P.intent = "item"
    elif P.patch or P.patch_unknown or P.why or any(w in q for w in PATCH_WORDS):
        P.intent = "patch"
    elif P.role:
        P.intent = "role"
    elif stat_word and not P.slot and P.champion:
        P.intent, P.stat = "stats", STAT_WORDS[stat_word]
    elif P.champion:
        P.intent = "skill"
    P.keywords = keywords(re.sub(r"(?<![A-Za-z])[QWERqwer](?![A-Za-z])", " ", rest))
    if P.intent == "patch" and P.champion and not P.patch and P.champion in BY_CHAMP:
        # 패치를 안 적은 경우: '질문한 항목(슬롯·키워드)이 마지막으로 바뀐 패치'
        def hits(r):
            lines = [c for c in r["changes"] if not P.slot or c["slot"] == P.slot]
            kw = sum(any(k in c["text"] for c in lines) for k in P.keywords)
            return bool(lines), kw
        recs = BY_CHAMP[P.champion]
        slot_recs = [r for r in recs if hits(r)[0]] or recs
        best_kw = max((hits(r)[1] for r in slot_recs), default=0)
        kw_recs = [r for r in slot_recs if best_kw and hits(r)[1] == best_kw]
        P.patch, P.patch_auto = (kw_recs or slot_recs)[0]["patch"], True
    return P


# ---------------- 검색 ----------------
def _where(conds):
    conds = [c for c in conds.values() if c]
    return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}


def search(q, mode=None, n=8):
    P = parse(q) if isinstance(q, str) else q
    conds = {"mode": {"mode": {"$in": [mode, "공통"]}} if mode else None}
    if P.intent == "patch":
        conds["type"] = {"type": {"$in": ["patch_context", "patch_change"]}} if P.why else {"type": "patch_change"}
        if P.patch:
            n = 60                                             # 한 (패치, 챔피언)의 변경 줄을 모두 가져오기
    elif P.intent == "skill":
        conds["type"] = {"type": "skill"} if P.slot else {"type": {"$in": ["skill", "profile", "tips"]}}
    elif P.intent == "stats":
        conds["type"] = {"type": "stats"}
    elif P.intent == "item":
        conds["type"] = {"type": "item"}
        conds["item"] = {"name": P.item}
        n = 5
    elif P.intent == "build":
        conds["type"] = {"type": "profile"}
    elif P.intent == "role":
        conds["type"] = {"type": "profile"}
        conds["role"] = {f"role_{P.role}": True}
        n = 200
    conds["champion"] = {"champion": P.champion} if P.champion else None
    conds["patch"] = {"patch": P.patch} if P.intent == "patch" and P.patch else None
    conds["slot"] = {"slot": P.slot} if P.slot and P.intent in ("patch", "skill") else None

    relaxed = []
    for drop in [None, "slot", "patch", "item", "type", "champion"]:
        if drop:
            if not conds.get(drop):
                continue
            conds[drop] = None
            relaxed.append(drop)
        res = collection().query(query_texts=[P.query], n_results=n, where=_where(conds))
        hits = [{"text": d, "meta": m, "dist": round(x, 4)}
                for d, m, x in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        if hits:
            return P, hits, relaxed
    return P, [], relaxed


# ---------------- 근거 선택 (규칙 기반, 교체 가능) ----------------
def rule_select(P, hits):
    def score(i, h):
        m, s = h["meta"], 0.0
        s += 3 * (P.champion is not None and m.get("champion") == P.champion)
        s += 2 * (P.slot is not None and m.get("slot") == P.slot)
        s += 2 * (P.patch is not None and m.get("patch") == P.patch)
        s += sum(1 for k in P.keywords if k in h["text"])
        s += 1 * (P.why and m.get("type") == "patch_context")
        return s - i * 1e-3                                    # 동점이면 벡터 유사도 순
    return max(range(len(hits)), key=lambda i: score(i, hits[i])) if hits else None


_SELECTOR = rule_select


def set_selector(fn):
    """fn(P, hits) -> 선택한 hits 인덱스. MiniGPT 리랭커를 붙일 때 사용"""
    global _SELECTOR
    _SELECTOR = fn


def select_line(P, hits):
    return _SELECTOR(P, hits)


# ---------------- 응답 ----------------
def _fmt_list(xs):
    return "/".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in xs)


def answer(q, mode=None):
    P, hits, relaxed = search(q, mode)
    best = select_line(P, hits)
    out = {"parsed": P, "hits": hits, "best": best, "relaxed": relaxed, "sources": [], "icon": None, "item": None, "build": None}
    if best is None:
        out["answer"] = "관련 정보를 찾지 못했습니다."
        return out
    h = hits[best]
    m = h["meta"]
    notes = []
    if P.patch_unknown:
        notes.append(f"{P.patch_unknown} 패치 데이터는 없습니다 (수집 범위: 13.1~최신).")
    if relaxed:
        notes.append(f"조건을 완화해 검색했습니다 (제외한 조건: {', '.join(relaxed)}).")

    t = m.get("type")
    if P.intent == "build":                                   # 템트리 추천 (recommend.py)
        import recommend as R
        r = R.recommend(P.query)
        out["answer"] = R.to_markdown(r)
        out["build"] = r if r["ok"] else None
        return out
    if t == "item" and m.get("item_id") in ITEM_BY_ID:
        it = ITEM_BY_ID[m["item_id"]]
        g = it["gold"]
        comp = [ITEM_BY_ID.get(x["id"], {"name": x["name"], "icon": None, "gold": {"total": None}}) for x in it["from"]]
        lines = [f"**{it['name']}** · {g['total']}골드 (조합 비용 {g['base']}, 판매 {g['sell']})"]
        if it["plaintext"]:
            lines.append(f"_{it['plaintext']}_")
        if it["stats"]:
            lines.append(f"- 능력치: {it['stats']}")
        if it["effect"]:
            lines.append(f"- 효과: {it['effect'][:400]}")
        if comp:
            lines.append("- 조합 재료: " + " + ".join(f"{c['name']}({c['gold']['total']}골드)" for c in comp))
        if it["into"]:
            lines.append("- 상위 아이템: " + ", ".join(x["name"] for x in it["into"][:6]))
        lines.append(f"- 사용 가능 모드: {', '.join(it['modes']) or '정보 없음'}")
        out["answer"] = "\n".join(lines)
        out["icon"] = it["icon"]
        out["item"] = {"name": it["name"], "icon": it["icon"], "gold": g["total"],
                       "components": [{"name": c["name"], "icon": c.get("icon"), "gold": c["gold"]["total"]} for c in comp]}
        notes.append(f"Data Dragon {it['version']} 기준")
        if any(w in P.query for w in PATCH_WORDS):
            notes.append("아이템 패치 변경 이력은 아직 수집하지 않았습니다.")
    elif t in ("patch_change", "patch_context"):
        rec = RECORDS[m["record"]]
        lines = rec["changes"]
        if P.slot and "slot" not in relaxed:
            lines = [c for c in lines if c["slot"] == P.slot] or lines
        kw = [c for c in lines if any(k in c["text"] for k in P.keywords)]
        lines = kw or lines
        head = f"**{rec['champion']} {patch_label(rec['patch'])} 패치**"
        if P.patch_auto and rec["patch"] == P.patch:
            head += f" (패치를 지정하지 않아 {rec['champion']}의 가장 최근 변경 패치 기준)"
        body = "\n".join(f"- {c['text']}" for c in lines)
        why = ""
        if (P.why or t == "patch_context") and (rec["summary"] or rec["context"]):
            why = "\n\n**변경 의도 (패치노트 개발자 코멘트)**\n> " + " ".join(x for x in (rec["summary"], rec["context"]) if x)
        out["answer"] = f"{head}\n{body}{why}"
        out["sources"] = [rec["url"]]
    elif t == "skill":
        c = CHAMPS[m["champion"]]
        s = next(s for s in c["skills"] if s["slot"] == m["slot"])
        extra = ""
        if s.get("cooldown") and any(s["cooldown"]):
            extra += f"\n- 재사용 대기시간: {_fmt_list(s['cooldown'])}초"
        if s.get("cost") and any(s["cost"]):
            extra += f"\n- 소모값: {_fmt_list(s['cost'])}"
        out["answer"] = f"**{c['name']} {SLOT_LABEL[s['slot']]} - {s['name']}**\n{s['description']}{extra}"
        out["icon"] = s["icon"]
        notes.append(f"Data Dragon {DD_VERSION} 기준")
    elif t == "stats":
        c = CHAMPS[m["champion"]]
        st = c["stats"]
        keys = [P.stat] if P.stat else list(STAT_NAME)
        rows = []
        for k in keys:
            per = st.get(f"{k}perlevel")
            rows.append(f"- {STAT_NAME[k]}: {st[k]:g}" + (f" (레벨당 +{per:g})" if per else ""))
        out["answer"] = f"**{c['name']} 기본 능력치**\n" + "\n".join(rows)
        out["icon"] = c["icon"]
        notes.append(f"Data Dragon {DD_VERSION} 기준")
    elif t == "profile" and P.intent == "role":
        names = sorted({x["meta"]["champion"] for x in hits})
        out["answer"] = (f"**{ROLE_KO[P.role]} 역할군 챔피언 {len(names)}명** (Data Dragon 분류 기준)\n"
                         + ", ".join(names)
                         + "\n\n※ 현재 메타·승률 기반 추천은 아직 지원하지 않습니다.")
    else:
        out["answer"] = h["text"]
        out["icon"] = m.get("icon")
    if notes:
        out["answer"] += "\n\n" + " ".join(f"_{x}_" for x in notes)
    return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "아트록스 26.19 패치 W 재사용 대기시간 어떻게 바뀌었어?"
    r = answer(q)
    print(r["parsed"])
    print(r["answer"])
    print("출처:", r["sources"])