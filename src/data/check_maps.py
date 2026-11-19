# src/data/check_maps.py
"""맵 ID 확인: Data Dragon map.json의 맵 이름 + 아이템 겹침 정도
실행 (프로젝트 루트에서):  /usr/local/bin/python3.10 src/data/check_maps.py
"""
import json
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DD = "https://ddragon.leagueoflegends.com"
ver = requests.get(f"{DD}/api/versions.json", timeout=15).json()[0]
raw = json.loads((ROOT / "data" / "raw" / "ddragon" / ver / "item.json").read_text(encoding="utf-8"))["data"]
items = [json.loads(l) for l in (ROOT / "data" / "processed" / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"버전 {ver}")
try:
    maps = requests.get(f"{DD}/cdn/{ver}/data/ko_KR/map.json", timeout=15).json()["data"]
    print("[Data Dragon map.json]")
    for k, v in sorted(maps.items(), key=lambda x: int(x[0])):
        print(f"  {k}: {v.get('MapName')}")
except Exception as e:
    print("map.json 없음/오류:", e)

keys = Counter(k for it in raw.values() for k in it.get("maps", {}))
print("\n[item.json에 나오는 맵 키 (true/false 포함)]", dict(sorted(keys.items(), key=lambda x: int(x[0]))))

on = lambda m: {it["id"] for it in items if m in it["maps"]}
s11, s12, s30, s453 = on("11"), on("12"), on("30"), on("453")
print(f"\n[겹침] 11:{len(s11)} 453:{len(s453)} | 11∩453 {len(s11 & s453)} | 453∩12 {len(s453 & s12)} | 453∩30 {len(s453 & s30)}")
print(f"  453에만 있고 11에는 없는 아이템 {len(s453 - s11)}개 (예: {[raw[i]['name'] for i in list(s453 - s11)[:10]]})")
print(f"  11에만 있고 453에는 없는 아이템 {len(s11 - s453)}개 (예: {[raw[i]['name'] for i in list(s11 - s453)[:10]]})")

print("\n[대표 아이템이 어느 맵에 있는지]")
for name in ["체력 물약", "시야 와드", "도란의 검", "장화", "삼위일체", "무한의 대검", "새끼 화염발톱"]:
    rows = [(i, sorted(m for m, v in it.get("maps", {}).items() if v), it.get("gold", {}).get("purchasable"))
            for i, it in raw.items() if it["name"] == name]
    print(f"  {name}: {rows}")