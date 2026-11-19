import requests
from bs4 import BeautifulSoup
import time

def get_patch_note_urls(list_url, max_patches=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(list_url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    links = soup.find_all("a", href=True)
    patch_urls = []
    seen = set()

    for link in links:
        href = link["href"]
        if "patch-" in href and "-notes" in href and href not in seen:
            full_url = href if href.startswith("http") else f"https://www.leagueoflegends.com{href}"
            patch_urls.append(full_url)
            seen.add(href)

    return patch_urls[:max_patches]


def crawl_patch_notes(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    start = soup.find(id="patch-champions")
    if not start:
        print(f"  ⚠️ {url} — patch-champions 영역을 찾지 못했습니다. 건너뜁니다.")
        return []

    elements = []
    for el in start.find_all_next(["h2", "a", "h4", "li"]):
        if el.name == "h2":
            break
        elements.append(el)

    structured_data = []
    current_champion = None
    current_skill = None

    # 패치 버전 번호를 URL에서 추출 (예: 26-19)
    patch_version = url.split("patch-")[-1].replace("-notes", "")

    for el in elements:
        if el.name == "a":
            text = el.get_text(strip=True)
            href = el.get("href", "")
            if text and len(text) <= 15 and ("/champion" in href or "/ko-kr/champions" in href):
                current_champion = text
                current_skill = None

        elif el.name == "h4":
            text = el.get_text(strip=True)
            if text:
                current_skill = text

        elif el.name == "li":
            text = el.get_text(strip=True)
            if text and current_champion:
                structured_data.append({
                    "champion": current_champion,
                    "skill": current_skill if current_skill else "일반",
                    "change": text,
                    "patch": patch_version,
                })

    return structured_data


# === 메인 실행 ===
list_url = "https://www.leagueoflegends.com/ko-kr/news/tags/patch-notes/"
patch_urls = get_patch_note_urls(list_url, max_patches=5)

print(f"총 {len(patch_urls)}개 패치노트 크롤링 시작\n")

all_data = []
for url in patch_urls:
    print(f"크롤링 중: {url}")
    data = crawl_patch_notes(url)
    print(f"  → {len(data)}건 수집")
    all_data.extend(data)
    time.sleep(1)  # 서버 부담 방지

print(f"\n전체 패치 통합 결과: 총 {len(all_data)}건")

# 패치별 집계
from collections import Counter
patch_counts = Counter(d["patch"] for d in all_data)
print("\n=== 패치별 수집 건수 ===")
for patch, cnt in sorted(patch_counts.items()):
    print(f"{patch}: {cnt}건")

# 저장 — 패치 버전 정보 포함
with open("../../data/raw/crawled_data.txt", "w", encoding="utf-8") as f:
    for d in all_data:
        f.write(f"[patch_{d['patch']}][{d['champion']}] {d['skill']}: {d['change']}\n")

print("\ncrawled_data.txt 저장 완료")