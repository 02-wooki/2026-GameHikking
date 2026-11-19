# src/data/aggregate_matches.py
"""A안 매치 데이터 첫 집계: data/raw/riot/participants.jsonl -> data/processed/build_stats.json
- 챔피언 x 포지션: 경기 수, 승률, 완성 아이템 등장률·승률 상위, 신발
- 챔피언 x 포지션 x 상대 라이너: 경기 수, 승률 (표본 적은 조합은 제외)
- '완성 아이템' = items.jsonl(소환사의 협곡)에서 상위 아이템이 없고 가격 2,000골드 이상인 아이템
※ 수집 대상이 한국 서버 마스터 이상이므로 '상위권 기준' 통계임

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/data/aggregate_matches.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PART = ROOT / "data" / "raw" / "riot" / "participants.jsonl"
ITEMS = ROOT / "data" / "processed" / "items.jsonl"
CHAMPS = ROOT / "data" / "processed" / "champions.jsonl"
OUT = ROOT / "data" / "processed" / "build_stats.json"
MIN_GAMES = 20          # 챔피언x포지션 최소 경기 수
MIN_MATCHUP = 10        # 상대 라이너 조합 최소 경기 수
TOP_ITEMS = 8


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


items = {it["id"]: it for it in load(ITEMS) if "소환사의 협곡" in it["modes"]}
legend = {i for i, it in items.items() if not it["into"] and (it["gold"]["total"] or 0) >= 2000}
boots = {i for i, it in items.items() if "Boots" in it["tags"] and i != "1001"}   # 기본 장화 제외, 2·3단계 신발 모두
en2ko = {c["id"].lower(): c["name"] for c in load(CHAMPS)}   # FiddleSticks/Fiddlesticks 대소문자 차이

rows = load(PART)
ver = Counter(".".join((r["version"] or "").split(".")[:2]) for r in rows)
print(f"선수 기록 {len(rows):,}건 | 매치 {len({r['match_id'] for r in rows}):,}개")
print("게임 버전 분포:", dict(ver.most_common(5)))

g = defaultdict(lambda: {"n": 0, "w": 0, "items": Counter(), "iw": Counter(), "boots": Counter()})
mu = defaultdict(lambda: [0, 0])
unknown = Counter()
for r in rows:
    c = en2ko.get((r["champion"] or "").lower())
    pos = r["position"]
    if not c or not pos:
        if not c:
            unknown[r["champion"]] += 1
        continue
    s = g[(c, pos)]
    s["n"] += 1
    s["w"] += bool(r["win"])
    for i in {str(x) for x in r["items"] if x}:
        if i in legend:
            s["items"][i] += 1
            s["iw"][i] += bool(r["win"])
        elif i in boots:
            s["boots"][i] += 1
    o = en2ko.get((r["opponent"] or "").lower())
    if o:
        mu[(c, pos, o)][0] += 1
        mu[(c, pos, o)][1] += bool(r["win"])

out = {"source": "Riot API, KR 마스터 이상 솔로랭크", "players": len(rows), "versions": dict(ver), "champions": {}}
for (c, pos), s in g.items():
    if s["n"] < MIN_GAMES:
        continue
    top = [{"id": i, "name": items[i]["name"], "pick": round(n / s["n"], 3), "win": round(s["iw"][i] / n, 3)}
           for i, n in s["items"].most_common(TOP_ITEMS)]
    bt = [{"id": i, "name": items[i]["name"], "pick": round(n / s["n"], 3)} for i, n in s["boots"].most_common(3)]
    vs = sorted(({"opponent": o, "games": n, "win": round(w / n, 3)}
                 for (cc, pp, o), (n, w) in mu.items() if cc == c and pp == pos and n >= MIN_MATCHUP),
                key=lambda x: -x["games"])
    out["champions"].setdefault(c, {})[pos] = {"games": s["n"], "win": round(s["w"] / s["n"], 3),
                                               "items": top, "boots": bt, "matchups": vs}

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
n_cp = sum(len(v) for v in out["champions"].values())
print(f"\n저장: {OUT}")
print(f"  챔피언x포지션 {n_cp}개 (경기 {MIN_GAMES}회 이상) | 챔피언 {len(out['champions'])}명")
print(f"  상대 라이너 조합(경기 {MIN_MATCHUP}회 이상): {sum(len(p['matchups']) for v in out['champions'].values() for p in v.values())}개")
print(f"  이름 매칭 실패 챔피언: {dict(unknown.most_common(5))}")
for name, pos in [("나서스", "TOP"), ("다리우스", "TOP")]:
    p = out["champions"].get(name, {}).get(pos)
    if p:
        print(f"\n--- {name} {pos}: {p['games']}경기, 승률 {p['win'] * 100:.1f}%")
        print("  완성 아이템:", [f"{x['name']} {x['pick'] * 100:.0f}%" for x in p["items"][:5]])
        print("  신발:", [f"{x['name']} {x['pick'] * 100:.0f}%" for x in p["boots"]])
        print("  상대:", [f"{x['opponent']} {x['games']}경기 {x['win'] * 100:.0f}%" for x in p["matchups"][:5]])