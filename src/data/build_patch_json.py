# src/build_patch_json.py
"""패치노트 캐시 HTML -> 구조화 JSONL (서버 요청 없음)
파이프라인: crawler_all.py (raw: data/raw/patch_html/*.html) -> 이 스크립트 (processed: data/processed/patch_notes.jsonl)

레코드 1개 = (패치, 챔피언) 1개:
{
  "id": "26-19_아트록스", "patch": "26-19", "patch_order": 261900,
  "url": "https://www.leagueoflegends.com/ko-kr/news/game-updates/...",
  "champion": "아트록스",
  "summary": "W 재사용 대기시간이 감소합니다.",          # 패치노트의 한 줄 요약 (없으면 null)
  "context": "아트록스가 ... 판단했습니다.",              # 개발자 코멘트 (없으면 null)
  "changes": [
    {"slot": "W", "skill": "지옥사슬", "field": "재사용 대기시간",
     "before": "20/18/16/14/12초", "after": "18/16.5/15/13.5/12초",
     "text": "W - 지옥사슬: 재사용 대기시간: 20/18/16/14/12초 ⇒ 18/16.5/15/13.5/12초"}
  ]
}
- slot: Q/W/E/R/패시브 또는 h4 제목 그대로(예: 기본 능력치), 소제목이 없으면 '일반'
- before/after: '⇒'가 없는 변경(신규 효과, 설명 변경 등)은 null
- patch_order: 정렬용 정수 (25-S1-x 시즌 패치는 같은 해 번호 패치보다 앞)

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/build_patch_json.py
"""
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "patch_html"
OUT_DIR = ROOT / "data" / "processed"
OUT = OUT_DIR / "patch_notes.jsonl"

CHAMPS = set(re.findall(r"^=== (.+?) ===$", (RAW / "champion_base_data.txt").read_text(encoding="utf-8"), re.M))
JUNK = ("트레일러", "시네마틱", "배경 이야기", "기획 해설")
SKILL_RE = re.compile(r"^(기본 지속 효과|[QWER])\s*-\s*(.+)$")


def norm(t):
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+:", ":", t)          # "기본 방어력 : 18" -> "기본 방어력: 18"


def patch_order(ver):
    m = re.match(r"^(\d+)-S(\d+)-(\d+)$", ver)
    if m:
        return int(m[1]) * 10000 + int(m[3])
    y, n = ver.split("-")
    return int(y) * 10000 + int(n) * 100


def split_skill(title):
    title = title.lstrip("- ").strip() if title else title
    if not title:
        return "일반", None
    m = SKILL_RE.match(title)
    if m:
        return ("패시브" if m[1] == "기본 지속 효과" else m[1]), m[2].strip()
    return title, None


def own_text(li):
    """중첩 li의 글자는 빼고 이 li에 직접 속한 글자만 (중첩 항목은 따로 수집되므로 중복 방지)"""
    return norm(" ".join(s for s in li.find_all(string=True) if s.find_parent("li") is li))


def parse_change(li, slot, skill, title):
    text = own_text(li)
    if not text or any(j in text for j in JUNK):
        return None
    strong = li.find("strong")
    field = norm(strong.get_text(" ")) if strong and strong.find_parent("li") is li else None
    if field and text.startswith(field):
        value = text[len(field):].lstrip(" :").strip()
    elif ":" in text:
        field, value = (norm(x) for x in text.split(":", 1))
    else:
        field, value = None, text
    before = after = None
    if "⇒" in value:
        before, after = (norm(x) for x in value.split("⇒", 1))
    return {"slot": slot, "skill": skill, "field": field, "before": before, "after": after,
            "text": f"{title or '일반'}: {text}"}


def find_start(soup):
    start = soup.find(id="patch-champions")
    if start:
        return start
    return next((h for h in soup.find_all("h2") if h.get_text(strip=True) == "챔피언"), None)


def canonical_url(soup):
    t = soup.find("link", rel="canonical")
    if t and t.get("href"):
        return t["href"]
    t = soup.find("meta", property="og:url")
    return t.get("content") if t else None


def parse(html, ver):
    soup = BeautifulSoup(html, "html.parser")
    start = find_start(soup)
    if not start:
        return []
    url = canonical_url(soup)
    by_champ, cur, title, slot, skill = {}, None, None, "일반", None
    for el in start.find_all_next(["h2", "a", "h4", "li", "p", "blockquote"]):
        if el.name == "h2":
            break
        if el.name == "a":
            t = norm(el.get_text(" "))
            href = el.get("href", "")
            # 변경 문장·코멘트 속 챔피언 링크는 무시 (챔피언 제목 링크만 인정)
            if (t in CHAMPS and ("/champion" in href or "/ko-kr/champions" in href)
                    and not el.find_parent(["li", "p", "blockquote"])):
                if t not in by_champ:
                    by_champ[t] = {"id": f"{ver}_{t}", "patch": ver, "patch_order": patch_order(ver),
                                   "url": url, "champion": t, "summary": None, "context": None, "changes": []}
                cur, title, slot, skill = by_champ[t], None, "일반", None
            continue
        if cur is None:
            continue
        if el.name == "p" and "summary" in (el.get("class") or []):
            cur["summary"] = cur["summary"] or norm(el.get_text(" ")) or None
        elif el.name == "blockquote":
            t = norm(el.get_text(" "))
            if t:
                cur["context"] = f"{cur['context']} {t}" if cur["context"] else t
        elif el.name == "h4":
            title = norm(el.get_text(" ")) or None
            slot, skill = split_skill(title)
        elif el.name == "li":
            c = parse_change(el, slot, skill, title)
            if c:
                cur["changes"].append(c)
    return [r for r in by_champ.values() if r["changes"]]


files = sorted(CACHE.glob("*.html"), key=lambda f: patch_order(f.stem))
records = []
for f in files:
    records += parse(f.read_text(encoding="utf-8"), f.stem)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as fo:
    for r in records:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

changes = [c for r in records for c in r["changes"]]
print(f"패치 {len(files)}개 -> (패치, 챔피언) 레코드 {len(records)}개 | 변경 {len(changes)}건 "
      f"| 챔피언 {len({r['champion'] for r in records})}명")
print(f"  수치 변경(before/after 있음): {sum(c['before'] is not None for c in changes)}건")
print(f"  field 없음: {sum(c['field'] is None for c in changes)}건")
print(f"  요약 있음: {sum(r['summary'] is not None for r in records)}개 | "
      f"개발자 코멘트 있음: {sum(r['context'] is not None for r in records)}개 "
      f"({sum(len(r['context'] or '') for r in records):,}자)")
print(f"  URL 없음: {sum(r['url'] is None for r in records)}개")
print("  slot 분포:", dict(Counter(c["slot"] for c in changes).most_common(10)))
print(f"  참고: crawled_data_all.txt 4,199줄과의 차이는 중첩 항목 중복 제거·문장 속 링크 오인 제거 때문")

ex = next((r for r in records if r["context"] and len(r["changes"]) >= 2), records[0])
show = dict(ex, changes=ex["changes"][:3])
print("\n--- 레코드 예시 ---")
print(json.dumps(show, ensure_ascii=False, indent=2))
print(f"\n저장: {OUT}")