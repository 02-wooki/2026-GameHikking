# src/data/build_items_json.py
"""Data Dragon 아이템 -> data/processed/items.jsonl (+ 맵 ID <-> 모드 대응 검증 출력)
raw 캐시: data/raw/ddragon/<버전>/item.json

레코드 예:
{"id": "3078", "name": "삼위일체", "plaintext": "...", "stats": "공격력 +36, ...", "effect": "...",
 "gold": {"base": 333, "total": 3333, "sell": 2333}, "from": [{"id": "3057", "name": "광휘의 검"}, ...],
 "into": [...], "tags": ["Damage", ...], "maps": ["11", "12"], "modes": ["소환사의 협곡", "칼바람"],
 "icon": "https://ddragon.../img/item/3078.png", "version": "..."}

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/data/build_items_json.py
출력의 [맵 ID 검증]을 보고 MAP_MODES가 맞는지 확인할 것 (각 맵에만 있는 아이템 이름으로 판단).
"""
import json
import re
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ddragon"
OUT = ROOT / "data" / "processed" / "items.jsonl"
DD = "https://ddragon.leagueoflegends.com"

# 검증 대상 (collect_lol_data.py와 같은 가정). 출력의 '그 맵에만 있는 아이템'으로 확인한다.
MAP_MODES = {"11": "소환사의 협곡", "12": "칼바람", "30": "아레나"}


def clean(t):
    t = re.sub(r"<br\s*/?>", " ", t or "")
    t = re.sub(r"</?[A-Za-z][^>]*>", "", t)
    return re.sub(r"\s+", " ", t).strip()


def split_desc(desc):
    """description = <mainText><stats>능력치</stats> 효과...</mainText>"""
    m = re.search(r"<stats>(.*?)</stats>", desc or "", re.S)
    stats = ""
    if m:
        stats = ", ".join(x for x in (clean(s) for s in re.split(r"<br\s*/?>", m.group(1))) if x)
    effect = clean(re.sub(r"<stats>.*?</stats>", " ", desc or "", flags=re.S))
    return stats, effect


def main():
    ver = requests.get(f"{DD}/api/versions.json", timeout=15).json()[0]
    cache = RAW / ver / "item.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        res = requests.get(f"{DD}/cdn/{ver}/data/ko_KR/item.json", timeout=15)
        res.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(res.text, encoding="utf-8")
        raw = res.json()
    data = raw["data"]
    name_of = {k: v["name"] for k, v in data.items()}
    print(f"Data Dragon {ver} | 아이템 원본 {len(data)}개")

    records, skipped = [], Counter()
    for iid, it in data.items():
        g = it.get("gold", {})
        if not g.get("purchasable"):
            skipped["구매 불가"] += 1
            continue
        if it.get("inStore") is False:
            skipped["상점 미노출"] += 1
            continue
        if it.get("requiredChampion") or it.get("requiredAlly"):
            skipped["특정 챔피언 전용"] += 1
            continue
        maps = sorted(m for m, v in it.get("maps", {}).items() if v)
        stats, effect = split_desc(it.get("description"))
        records.append({
            "id": iid, "name": it["name"], "plaintext": clean(it.get("plaintext")),
            "stats": stats, "effect": effect,
            "gold": {"base": g.get("base"), "total": g.get("total"), "sell": g.get("sell")},
            "from": [{"id": i, "name": name_of.get(i, i)} for i in it.get("from", [])],
            "into": [{"id": i, "name": name_of.get(i, i)} for i in it.get("into", [])],
            "tags": it.get("tags", []), "depth": it.get("depth", 1),
            "maps": maps, "modes": [MAP_MODES[m] for m in maps if m in MAP_MODES],
            "icon": f"{DD}/cdn/{ver}/img/item/{it['image']['full']}", "version": ver,
        })

    # ---------- 맵 ID 검증 ----------
    print("\n[맵 ID 검증] 맵마다 아이템 수, 그 맵에만 있는 아이템 예시")
    per_map = Counter(m for r in records for m in r["maps"])
    for m, n in sorted(per_map.items(), key=lambda x: int(x[0])):
        only = [r["name"] for r in records if r["maps"] == [m]]
        print(f"  맵 {m:>3}: {n:4d}개 -> {MAP_MODES.get(m, '미분류(모드 없음)'):8s} | 이 맵 전용 {len(only)}개: {only[:8]}")
    both = [r["name"] for r in records if {"11", "12"} <= set(r["maps"])]
    print(f"  맵 11과 12 모두: {len(both)}개 (예: {both[:5]})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    names = Counter(r["name"] for r in records)
    dup = {n: c for n, c in names.items() if c > 1}
    print(f"\n저장: {OUT}")
    print(f"  구매 가능 아이템 {len(records)}개 | 제외 {dict(skipped)}")
    print(f"  모드 분포: {dict(Counter(m for r in records for m in r['modes']))}")
    print(f"  모드가 하나도 없는 아이템: {sum(not r['modes'] for r in records)}개")
    print(f"  이름이 겹치는 아이템: {len(dup)}종 (예: {list(dup.items())[:5]})")
    ex = next((r for r in records if r["name"] == "삼위일체"), records[0])
    print("\n--- 레코드 예시 ---")
    print(json.dumps(ex, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()