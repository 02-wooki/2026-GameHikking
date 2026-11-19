import requests
import json

# 1. 최신 패치 버전 확인
versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
latest_version = versions[0]
print(f"최신 버전: {latest_version}")

# 2. 전체 챔피언 목록 가져오기
champ_list_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/ko_KR/champion.json"
champ_list = requests.get(champ_list_url).json()["data"]
print(f"전체 챔피언 수: {len(champ_list)}명")

# 3. 각 챔피언의 상세 정보 (스킬 포함) 가져오기
champion_docs = []

for champ_key in champ_list:
    detail_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/ko_KR/champion/{champ_key}.json"
    detail = requests.get(detail_url).json()["data"][champ_key]

    name = detail["name"]
    title = detail["title"]
    tags = ", ".join(detail["tags"])  # 역할군 (예: Fighter, Tank)
    lore = detail["lore"]

    # 기본 스탯
    stats = detail["stats"]
    stat_text = f"기본 체력: {stats['hp']}, 공격력: {stats['attackdamage']}, 방어력: {stats['armor']}, 이동속도: {stats['movespeed']}"

    # 스킬 (기본 지속 효과 + Q/W/E/R)
    passive = detail["passive"]
    skill_lines = [f"기본 지속 효과 - {passive['name']}: {passive['description']}"]

    for i, spell in enumerate(detail["spells"]):
        key = ["Q", "W", "E", "R"][i]
        skill_lines.append(f"{key} - {spell['name']}: {spell['description']}")

    text = (
        f"{name} ({title})\n"
        f"역할군: {tags}\n"
        f"{stat_text}\n"
        + "\n".join(skill_lines)
    )

    champion_docs.append({"champion": name, "text": text})
    print(f"수집 완료: {name}")

# 4. 저장
with open("../../data/raw/champion_base_data.txt", "w", encoding="utf-8") as f:
    for doc in champion_docs:
        f.write(f"=== {doc['champion']} ===\n{doc['text']}\n\n")

print(f"\nchampion_base_data.txt 저장 완료 (총 {len(champion_docs)}명)")