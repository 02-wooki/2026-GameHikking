# src/crawler_all.py
"""전체 패치노트 수집 (13.1 ~ 현재)
목록 페이지는 최근 12개만 보여주고 나머지는 '더 보기'(자바스크립트)라서,
패치 번호로 URL 후보를 직접 만들어 존재하는 페이지만 수집한다.

확인된 URL 형식 (연도마다 다름):
  2023~2024: /ko-kr/news/game-updates/patch-14-24-notes/
  2025     : /ko-kr/news/game-updates/patch-25-16-notes/ , patch-25-s1-1-notes/
  2026     : /ko-kr/news/game-updates/league-of-legends-patch-26-19-notes

- 기존 crawled_data.txt는 건드리지 않는다 (SFT 홀드아웃 분할이 바뀌지 않게).
  결과는 data/raw/crawled_data_all.txt 에 따로 저장.
- 받은 HTML은 data/raw/patch_html/ 에 캐시 -> 다시 실행하면 서버에 재요청하지 않음.
- v2 수정: ① 기본정보 173명 명단에 있는 이름만 챔피언으로 인정 ('여기', '…배경 이야기' 등 링크 텍스트 제거)
           ② 챔피언 섹션 h2에 id가 없으면 제목이 '챔피언'인 h2를 시작점으로 사용 (14-05, 14-12, 25-05)
           ③ 트레일러·시네마틱 안내 줄 제외 ④ 5건 미만 패치는 챔피언 섹션 앞부분 태그를 출력 (구조 확인용)

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/crawler_all.py
"""
import time
from collections import Counter
from pathlib import Path

import re

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "patch_html"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = RAW / "crawled_data_all.txt"
CHAMPS = set(re.findall(r"^=== (.+?) ===$", (RAW / "champion_base_data.txt").read_text(encoding="utf-8"), re.M))
JUNK = ("트레일러", "시네마틱", "배경 이야기", "기획 해설")

BASE = "https://www.leagueoflegends.com/ko-kr/news/game-updates/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DELAY = 1.0                      # 요청 간격(초). 서버 부담 방지


def versions():
    """(저장용 패치 표기, URL에 쓸 번호 후보들)"""
    out = []
    for y in (13, 14):
        for n in range(1, 25):
            out.append((f"{y}-{n:02d}", [f"{y}-{n}"]))
    for n in range(1, 4):
        out.append((f"25-S1-{n}", [f"25-s1-{n}"]))
    for n in range(1, 25):
        out.append((f"25-{n:02d}", [f"25-{n:02d}", f"25-{n}"]))
    for n in range(1, 25):
        out.append((f"26-{n:02d}", [f"26-{n:02d}", f"26-{n}"]))
    return out


def candidates(nums):
    for num in nums:
        for slug in (f"league-of-legends-patch-{num}-notes", f"patch-{num}-notes"):
            yield BASE + slug
            yield BASE + slug + "/"


def fetch(ver, nums):
    """캐시가 있으면 캐시, 없으면 후보 URL을 차례로 시도. 실패하면 None."""
    cache = CACHE / f"{ver}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8"), "캐시"
    seen = set()
    for url in candidates(nums):
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
        except requests.RequestException as e:
            print(f"  ⚠️ {url} 요청 오류: {e}")
            time.sleep(DELAY)
            continue
        time.sleep(DELAY)
        if res.status_code == 200 and "patch-champions" in res.text:
            cache.write_text(res.text, encoding="utf-8")
            return res.text, url
        if res.status_code == 200:   # 페이지는 있는데 챔피언 섹션 id가 없음 (구조가 다른 패치)
            cache.write_text(res.text, encoding="utf-8")
            return res.text, url + " (챔피언 섹션 id 없음)"
    return None, None


def find_start(soup):
    start = soup.find(id="patch-champions")
    if start:
        return start
    return next((h for h in soup.find_all("h2") if h.get_text(strip=True) == "챔피언"), None)


def dump_section(html, n=40):
    """챔피언 섹션 앞부분 태그 출력 (구조 확인용)"""
    start = find_start(BeautifulSoup(html, "html.parser"))
    if not start:
        print("    (챔피언 섹션 없음)")
        return
    for i, el in enumerate(start.find_all_next()):
        if i >= n or el.name == "h2":
            break
        t = el.get_text(" ", strip=True)[:60]
        if t:
            print(f"    <{el.name} class={el.get('class')}> {t}")


def parse(html, ver):
    """기존 crawler.py와 같은 규칙: #patch-champions부터 다음 h2 전까지, a=챔피언 / h4=스킬 / li=변경"""
    soup = BeautifulSoup(html, "html.parser")
    start = find_start(soup)
    if not start:
        return []
    data, champ, skill = [], None, None
    for el in start.find_all_next(["h2", "a", "h4", "li"]):
        if el.name == "h2":
            break
        text = el.get_text(strip=True)
        if el.name == "a":
            href = el.get("href", "")
            if text in CHAMPS and ("/champion" in href or "/ko-kr/champions" in href):
                champ, skill = text, None
        elif el.name == "h4":
            if text:
                skill = text
        elif el.name == "li" and text and champ:
            if any(j in text for j in JUNK):
                continue
            data.append({"champion": champ, "skill": skill or "일반", "change": text, "patch": ver})
    return data


all_data, status = [], {}
for ver, nums in versions():
    html, src = fetch(ver, nums)
    if html is None:
        status[ver] = "없음"
        print(f"{ver}: 페이지 없음")
        continue
    d = parse(html, ver)
    status[ver] = len(d)
    print(f"{ver}: {len(d)}건  <- {src}")
    if len(d) < 5:
        print(f"  [구조 확인] {ver} 챔피언 섹션 앞부분:")
        dump_section(html)
    all_data.extend(d)

with OUT.open("w", encoding="utf-8") as f:
    for d in all_data:
        f.write(f"[patch_{d['patch']}][{d['champion']}] {d['skill']}: {d['change']}\n")

found = [v for v, s in status.items() if s != "없음"]
zero = [v for v, s in status.items() if s == 0]
print("\n===== 요약 =====")
print(f"페이지 찾음 {len(found)}개 / 못 찾음 {len(status) - len(found)}개")
print(f"페이지는 있는데 0건(구조가 다를 가능성): {zero}")
print(f"총 {len(all_data)}건 | 챔피언 {len({d['champion'] for d in all_data})}명 | "
      f"(챔피언, 패치) 청크 {len({(d['champion'], d['patch']) for d in all_data})}개")
print("변경 줄 중 '⇒' 포함(수치 변경):", sum('⇒' in d['change'] for d in all_data))
print(f"기본정보 명단에 없는 챔피언명: {sorted({d['champion'] for d in all_data} - CHAMPS)}")
print("\n챔피언명 상위 15 (이상한 이름이 섞였는지 확인):")
for c, n in Counter(d["champion"] for d in all_data).most_common(15):
    print(f"  {c}: {n}")
print(f"\n저장: {OUT}")