# collect_lol_data.py  (v2: 모드 태깅)
import json
import re
import requests
from collections import Counter
from pathlib import Path

BASE = "https://ddragon.leagueoflegends.com"
OUT = Path("../../data/chunks")
OUT.mkdir(parents=True, exist_ok=True)

# 맵 ID -> 모드. 실행 후 출력으로 반드시 검증할 것
MAP_MODES = {"11": "소환사의 협곡", "12": "칼바람", "30": "아레나"}
# 소환사 주문의 modes 값 -> 모드
SPELL_MODES = {"CLASSIC": "소환사의 협곡", "ARAM": "칼바람", "CHERRY": "아레나"}
COMMON = "공통"


def clean(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", t).strip()


ver = requests.get(f"{BASE}/api/versions.json", timeout=10).json()[0]
print("버전:", ver)

records, terms = [], set(MAP_MODES.values()) | {"증강 칼바람", "전략적 팀 전투"}


def add(text, mode, typ, name, icon=None):
    records.append({"text": f"[{mode}] {text}", "mode": mode,
                    "type": typ, "name": name, "icon": icon})


# 1. 아이템: 활성화된 맵마다 모드별 레코드 생성
items = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/item.json", timeout=10).json()["data"]
id2name = {k: v["name"] for k, v in items.items()}
map_counter = Counter()

for iid, it in items.items():
    if not it.get("gold", {}).get("purchasable", False):
        continue
    name = it["name"]
    terms.add(name)
    recipe = [id2name[i] for i in it.get("from", []) if i in id2name]
    icon = f"{BASE}/cdn/{ver}/img/item/{it['image']['full']}"
    body = f"아이템 {name}: {clean(it.get('description', ''))} 총 가격 {it['gold']['total']}골드."
    if recipe:
        body += f" 조합식: {' + '.join(recipe)}."
    for map_id, enabled in it.get("maps", {}).items():
        if not enabled:
            continue
        map_counter[map_id] += 1
        mode = MAP_MODES.get(map_id)
        if mode:
            add(body, mode, "item", name, icon)

print("\n[아이템] 맵 ID별 개수 (대응표에 없는 키는 저장 안 됨):")
for k, v in sorted(map_counter.items()):
    print(f"  맵 {k}: {v}개  -> {MAP_MODES.get(k, '미분류(저장 안 함)')}")

# 2. 소환사 주문
spells = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/summoner.json", timeout=10).json()["data"]
for sp in spells.values():
    terms.add(sp["name"])
    icon = f"{BASE}/cdn/{ver}/img/spell/{sp['image']['full']}"
    for m in sp.get("modes", []):
        if m in SPELL_MODES:
            add(f"소환사 주문 {sp['name']}: {clean(sp['description'])}",
                SPELL_MODES[m], "spell", sp["name"], icon)

# 3. 룬 (모드 구분 정보 없음 -> 공통)
runes = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/runesReforged.json", timeout=10).json()
for tree in runes:
    for slot in tree["slots"]:
        for r in slot["runes"]:
            terms.add(r["name"])
            add(f"룬 {r['name']} ({tree['name']}): {clean(r['longDesc'])}",
                COMMON, "rune", r["name"])

# 4. 챔피언 설정과 팁 (공통)
champs = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/champion.json", timeout=10).json()["data"]
for key in champs:
    d = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/champion/{key}.json",
                     timeout=10).json()["data"][key]
    name = d["name"]
    terms.add(name)
    add(f"{name} 설정: {clean(d['lore'])}", COMMON, "lore", name)
    for tip in d.get("allytips", []):
        add(f"{name} 플레이 팁: {clean(tip)}", COMMON, "ally_tip", name)
    for tip in d.get("enemytips", []):
        add(f"{name} 상대 팁: {clean(tip)}", COMMON, "enemy_tip", name)

Path("data/chunks/ddragon_extra.txt").unlink(missing_ok=True)   # 구버전 중복 방지
with open(OUT / "ddragon_extra.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

term_path = Path("../../data/lol_terms.txt")
old = set(term_path.read_text(encoding="utf-8").splitlines()) if term_path.exists() else set()
term_path.write_text("\n".join(sorted(old | terms)), encoding="utf-8")

print("\n[모드 x 종류별 레코드 수]")
for (mode, typ), n in sorted(Counter((r["mode"], r["type"]) for r in records).items()):
    print(f"  {mode:10s} {typ:10s} {n}")
print(f"\n총 {len(records)}건 / {sum(len(r['text']) for r in records):,}자, 용어 {len(old | terms)}개")