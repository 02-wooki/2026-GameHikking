# make_corpus.py
import re
from pathlib import Path

Path("../../data/chunks").mkdir(parents=True, exist_ok=True)

corpus_lines = []
terms = set()

# 1. 챔피언 기본 데이터: 챔피언명, 스킬명 추출 + 텍스트 정리
with open("../../data/raw/champion_base_data.txt", encoding="utf-8") as f:
    raw = f.read()

for block in raw.strip().split("=== ")[1:]:
    champion, body = block.split(" ===\n", 1)
    terms.add(champion.strip())
    body = re.sub(r"<br\s*/?>", " ", body)  # HTML 태그 제거
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        corpus_lines.append(line)
        # "Q - 현혹의 구슬: 설명", "기본 지속 효과 - 정기 흡수: 설명"에서 스킬명 추출
        m = re.match(r"^(?:[QWER]|기본 지속 효과) - ([^:]+):", line)
        if m:
            terms.add(m.group(1).strip())

# 2. 패치노트 청크: 챔피언명 추가 + 스킬명 추출
with open("../../data/raw/chunks.txt", encoding="utf-8") as f:
    raw = f.read()

for block in raw.strip().split("=== ")[1:]:
    header, body = block.split(" ===\n", 1)
    terms.add(header.split("|")[0].strip())
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        corpus_lines.append(line)
        m = re.match(r"^(?:[QWER]|기본 지속 효과) - ([^:]+):", line)
        if m:
            terms.add(m.group(1).strip())

with open("../../data/chunks/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus_lines))

with open("../../data/lol_terms.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(terms)))

print(f"코퍼스 {len(corpus_lines)}줄, 도메인 용어 {len(terms)}개 저장 완료")