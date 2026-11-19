# src/build_champions_json.py
"""Data Dragon 챔피언 정보 -> 구조화 JSONL
파이프라인: Data Dragon 원본 JSON (raw: data/raw/ddragon/<버전>/champion/*.json, 캐시)
          -> data/processed/champions.jsonl (챔피언 1명 = 1줄)

- 기존 champion_base_data.txt는 건드리지 않는다 (SFT 실험 재현용). 기존 champion_data.py도 실행하지 말 것
  (최신 버전으로 덮어쓰면 SFT 홀드아웃 분할·데이터가 바뀜).
- 캐시가 있으면 다시 받지 않는다. 최신 버전이 바뀌면 새 버전 폴더에 새로 받는다.

레코드 예:
{"id": "Aatrox", "key": "266", "name": "아트록스", "title": "다르킨의 검", "version": "...",
 "tags": ["Fighter"], "roles": ["전사"], "partype": "피의 샘",
 "stats": {"hp": 650, "hpperlevel": 114, "armor": 38, ...},     # Data Dragon stats 전체
 "skills": [{"slot": "패시브", "name": "...", "description": "...", "icon": "..."},
            {"slot": "Q", "id": "AatroxQ", "name": "...", "description": "...",
             "cooldown": [14, 12, 10, 8, 6], "cost": [0, 0, 0, 0, 0], "range": [25000, ...],
             "maxrank": 5, "icon": "..."}, ...],
 "icon": "...", "allytips": [...], "enemytips": [...], "lore": "..."}
※ 스킬 피해량·계수는 넣지 않는다 (Data Dragon의 tooltip/effect 값은 신뢰하기 어려움). 쿨다운·소모값·사거리만 사용.

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/build_champions_json.py
"""
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "champions.jsonl"
DD = "https://ddragon.leagueoflegends.com"
SLOTS = ["Q", "W", "E", "R"]
ROLE_KO = {"Fighter": "전사", "Tank": "탱커", "Mage": "마법사", "Assassin": "암살자",
           "Marksman": "원거리 딜러", "Support": "서포터"}


def clean(text):
    """Data Dragon 마크업 제거 (build_sft_data.clean과 같은 규칙)"""
    text = re.sub(r"<br\s*/?>", " ", text or "")
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_json(url, cache):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(res.text, encoding="utf-8")
    time.sleep(0.1)
    return res.json()


def build_record(d, ver):
    img = f"{DD}/cdn/{ver}/img"
    p = d["passive"]
    skills = [{"slot": "패시브", "name": p["name"], "description": clean(p["description"]),
               "icon": f"{img}/passive/{p['image']['full']}"}]
    for slot, s in zip(SLOTS, d["spells"]):
        skills.append({"slot": slot, "id": s["id"], "name": s["name"],
                       "description": clean(s["description"]),
                       "cooldown": s.get("cooldown"), "cost": s.get("cost"), "range": s.get("range"),
                       "maxrank": s.get("maxrank"), "icon": f"{img}/spell/{s['image']['full']}"})
    return {"id": d["id"], "key": d["key"], "name": d["name"], "title": d["title"], "version": ver,
            "tags": d["tags"], "roles": [ROLE_KO.get(t, t) for t in d["tags"]], "partype": d.get("partype"),
            "stats": d["stats"], "skills": skills, "icon": f"{img}/champion/{d['image']['full']}",
            "allytips": [clean(t) for t in d.get("allytips", [])],
            "enemytips": [clean(t) for t in d.get("enemytips", [])],
            "lore": clean(d.get("lore"))}


def main():
    ver = requests.get(f"{DD}/api/versions.json", timeout=15).json()[0]
    base = RAW / "ddragon" / ver
    print(f"Data Dragon 최신 버전: {ver}")
    lst = get_json(f"{DD}/cdn/{ver}/data/ko_KR/champion.json", base / "champion.json")["data"]

    records = []
    for i, key in enumerate(sorted(lst), 1):
        d = get_json(f"{DD}/cdn/{ver}/data/ko_KR/champion/{key}.json",
                     base / "champion" / f"{key}.json")["data"][key]
        records.append(build_record(d, ver))
        if i % 40 == 0:
            print(f"  {i}/{len(lst)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    old = set(re.findall(r"^=== (.+?) ===$", (RAW / "champion_base_data.txt").read_text(encoding="utf-8"), re.M))
    new = {r["name"] for r in records}
    print(f"\n챔피언 {len(records)}명 저장: {OUT}")
    print(f"  기존 champion_base_data.txt 대비 새로 생긴 챔피언: {sorted(new - old)}")
    print(f"  기존에 있었는데 없어진 이름: {sorted(old - new)}")
    print(f"  스킬 5개(패시브+QWER)가 아닌 챔피언: {[r['name'] for r in records if len(r['skills']) != 5]}")
    print(f"  역할 분포: { {k: sum(k in r['roles'] for r in records) for k in ROLE_KO.values()} }")

    ex = next((r for r in records if r["name"] == "아트록스"), records[0])
    show = dict(ex, skills=ex["skills"][:2], lore=ex["lore"][:60] + "…",
                allytips=ex["allytips"][:1], enemytips=ex["enemytips"][:1])
    print("\n--- 레코드 예시 ---")
    print(json.dumps(show, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()