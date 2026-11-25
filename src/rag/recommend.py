# src/rag/recommend.py
"""규칙 기반 템트리 추천 (C안) + 상위권 매치 통계 근거 (A안 데이터)
입력 예: "상대 다리우스, 나 나서스 탑 템트리 추천해줘"

1) 질문에서 내 챔피언·상대 챔피언·포지션 추출
2) 후보 = build_stats.json의 '내 챔피언 x 포지션' 완성 아이템 (상위권 구매율 5% 이상)
3) 상대 피해 성향(Data Dragon info.attack / info.magic)에 맞는 방어 능력치가 있으면 우선순위 가산
4) 신발도 같은 규칙 (물리 -> 판금 장화, 마법 -> 헤르메스의 발걸음, 후보에 있을 때만)
5) 구매 순서 = 첫 핵심 아이템 재료 -> 신발 -> 핵심 아이템 1·2·3
모든 근거는 데이터에서 계산 (구매율·승률은 KR 마스터 이상 기준)
"""
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
POS_WORDS = {"탑": "TOP", "정글": "JUNGLE", "미드": "MIDDLE", "원딜": "BOTTOM", "바텀": "BOTTOM", "봇": "BOTTOM",
             "서폿": "UTILITY", "서포터": "UTILITY"}
POS_KO = {"TOP": "탑", "JUNGLE": "정글", "MIDDLE": "미드", "BOTTOM": "원딜", "UTILITY": "서포터"}
MIN_PICK = 0.05
DEF_BONUS = 0.15


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@lru_cache(maxsize=1)
def data():
    champs = {c["name"]: c for c in _load(PROC / "champions.jsonl")}
    items = {it["id"]: it for it in _load(PROC / "items.jsonl") if "소환사의 협곡" in it["modes"]}
    stats = json.loads((PROC / "build_stats.json").read_text(encoding="utf-8"))
    ver = next(iter(champs.values()))["version"]
    info = {}
    for c in champs.values():
        f = ROOT / "data" / "raw" / "ddragon" / ver / "champion" / f"{c['id']}.json"
        if f.exists():
            info[c["name"]] = json.loads(f.read_text(encoding="utf-8"))["data"][c["id"]].get("info", {})
    return champs, items, stats, info


def damage_type(name):
    """Data Dragon info(0~10)로 물리/마법/혼합 판정, 없으면 역할군으로 추정"""
    champs, _, _, info = data()
    i = info.get(name)
    if i:
        a, m = i.get("attack", 0), i.get("magic", 0)
        kind = "혼합" if abs(a - m) <= 2 else ("물리" if a > m else "마법")
        return kind, f"Data Dragon 성향값 공격 {a} / 마법 {m}"
    tags = champs[name]["tags"]
    kind = "마법" if "Mage" in tags else "물리"
    return kind, f"역할군({', '.join(champs[name]['roles'])})으로 추정"


def parse(q):
    from search import normalize                       # 줄임말·붙여 쓴 이름 ('마이', '마스터이') -> 정식 이름
    q = normalize(q)
    champs, _, _, _ = data()
    found = []
    for n in sorted(champs, key=len, reverse=True):
        for m in re.finditer(re.escape(n), q):
            if not any(s <= m.start() < e for s, e, _ in found):
                found.append((m.start(), m.end(), n))
    found.sort()
    enemy = mine = None
    for s, e, n in found:
        before, after = q[max(0, s - 4):s], q[e:e + 4]
        if "상대" in before or "vs" in before.lower() or after.startswith(" 상대") or after.startswith("상대"):
            enemy = enemy or n
    rest = [n for _, _, n in found if n != enemy]
    mine = rest[0] if rest else None
    if mine is None and len(found) >= 2:
        mine = found[-1][2]
    pos = next((POS_WORDS[w] for w in sorted(POS_WORDS, key=len, reverse=True) if w in q), None)
    return mine, enemy, pos


def card(it, **extra):
    return dict({"id": it["id"], "name": it["name"], "icon": it["icon"], "gold": it["gold"]["total"]}, **extra)


def recommend(q):
    champs, items, stats, _ = data()
    mine, enemy, pos = parse(q)
    if not mine:
        return {"ok": False, "message": "내 챔피언을 찾지 못했습니다. 예: '상대 다리우스, 나 나서스 탑 템트리 추천해줘'"}
    by_pos = stats["champions"].get(mine, {})
    if not by_pos:
        return {"ok": False, "message": f"{mine}의 상위권 경기 데이터가 부족합니다 (20경기 미만)."}
    if pos not in by_pos:
        pos = max(by_pos, key=lambda p: by_pos[p]["games"])          # 가장 많이 플레이된 포지션
    st = by_pos[pos]
    notes = []

    kind = reason = None
    want = None
    if enemy:
        kind, reason = damage_type(enemy)
        want = {"물리": "방어력", "마법": "마법 저항력"}.get(kind)

    cands = []
    for x in st["items"]:
        if x["pick"] < MIN_PICK or x["id"] not in items:
            continue
        it = items[x["id"]]
        defensive = bool(want and want in it["stats"])
        score = x["pick"] + (DEF_BONUS if defensive else 0)
        why = f"상위권 {mine} {POS_KO.get(pos, pos)} 구매율 {x['pick'] * 100:.0f}% (그 판 승률 {x['win'] * 100:.0f}%)"
        if defensive:
            why += f" · 상대 {enemy}({kind} 피해) 대응 {want}"
        cands.append((score, it, x, why))
    cands.sort(key=lambda c: -c[0])
    core = cands[:3]
    if not core:
        return {"ok": False, "message": f"{mine} {POS_KO.get(pos, pos)}의 완성 아이템 데이터가 부족합니다."}

    boot = None
    pref = {"물리": "판금 장화", "마법": "헤르메스의 발걸음"}.get(kind)
    bl = [b for b in st["boots"] if b["id"] in items]
    if bl:
        b = next((b for b in bl if b["name"] == pref), bl[0])
        boot = card(items[b["id"]], why=f"상위권 구매율 {b['pick'] * 100:.0f}%"
                    + (f" · 상대 {kind} 피해 대응" if b["name"] == pref else ""))

    first = core[0][1]
    comps = [card(items[c["id"]], why=f"{first['name']} 재료") for c in first["from"] if c["id"] in items]
    order = comps + ([boot] if boot else []) + [card(it, why=why, pick=x["pick"]) for _, it, x, why in core]

    matchup = next((m for m in st["matchups"] if m["opponent"] == enemy), None) if enemy else None
    if enemy and not matchup:
        notes.append(f"{mine} vs {enemy} 조합은 상위권 경기가 10판 미만이라, 상대 대응은 성향 규칙으로 계산했습니다.")
    return {"ok": True, "mine": mine, "enemy": enemy, "position": pos, "games": st["games"], "win": st["win"],
            "enemy_kind": kind, "enemy_reason": reason, "matchup": matchup, "order": order,
            "core": [c[1]["name"] for c in core], "notes": notes, "source": stats.get("source")}


def to_markdown(r):
    if not r["ok"]:
        return r["message"]
    lines = [f"**{r['mine']} {POS_KO.get(r['position'], r['position'])} 추천 템트리**"
             + (f" (상대 {r['enemy']})" if r["enemy"] else "")]
    lines.append(f"_근거: {r['source']} · {r['mine']} {POS_KO.get(r['position'])} {r['games']}경기, 승률 {r['win'] * 100:.1f}%_")
    if r["enemy"]:
        lines.append(f"- 상대 {r['enemy']}: **{r['enemy_kind']} 피해** 위주 ({r['enemy_reason']})")
    if r["matchup"]:
        m = r["matchup"]
        lines.append(f"- 상위권 맞대결: {m['games']}경기, {r['mine']} 승률 {m['win'] * 100:.0f}%")
    lines.append("\n**구매 순서**")
    for i, c in enumerate(r["order"], 1):
        lines.append(f"{i}. {c['name']} ({c['gold']}골드) — {c['why']}")
    for n in r["notes"]:
        lines.append(f"\n_{n}_")
    lines.append("\n_규칙 기반 추천입니다. 구매율·승률은 상위권 경기의 상관관계이며 인과를 뜻하지 않습니다._")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "상대 다리우스, 나 나서스 탑 템트리 추천해줘"
    print(to_markdown(recommend(q)))