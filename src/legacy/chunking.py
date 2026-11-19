import re
import chromadb
from chromadb.utils import embedding_functions

# 1. champion_base_data.txt 읽어서 챔피언별로 분리
base_chunks = []
with open("../../data/raw/champion_base_data.txt", "r", encoding="utf-8") as f:
    raw = f.read()

blocks = raw.strip().split("=== ")[1:]
for block in blocks:
    champion, text = block.split(" ===\n", 1)
    champion = champion.strip()
    text = text.strip()
    # HTML 태그(<br> 등) 제거
    text = re.sub(r"<br\s*/?>", " ", text)
    base_chunks.append({"text": text, "champion": champion})

print(f"챔피언 기본 데이터 {len(base_chunks)}개 로드 완료")

# 2. 기존 Vector DB에 추가 (컬렉션 재생성 X, 그대로 add만)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="distiluse-base-multilingual-cased-v1"
)

client = chromadb.PersistentClient(path="../../chroma_db")
collection = client.get_or_create_collection(
    name="gamehikking",
    embedding_function=embedding_fn
)

# 3. 기본 데이터 추가 (type: "base" 메타데이터로 패치 데이터와 구분)
for i, chunk in enumerate(base_chunks):
    collection.add(
        documents=[chunk["text"]],
        metadatas=[{"champion": chunk["champion"], "type": "base_info", "patch": "N/A"}],
        ids=[f"base_{i}"]
    )

print(f"Vector DB에 기본 정보 {len(base_chunks)}건 추가 완료")

total_count = collection.count()
print(f"DB 전체 청크 수: {total_count}개 (패치노트 64개 + 기본정보 {len(base_chunks)}개)")

# 4. 검색 테스트 — 이번엔 "스킬/역할군" 같은 기본 정보 질문
test_queries = [
    "아리 궁극기 뭐야?",
    "탱커 챔피언 추천해줘",
]

# smart_search 함수 재사용 (지난 코드와 동일)
all_metadata = collection.get()["metadatas"]
known_champions = sorted(set(m["champion"] for m in all_metadata if m.get("champion")), key=len, reverse=True)

def smart_search(query, n_results=3):
    matched_champion = None
    for champ in known_champions:
        if champ in query:
            matched_champion = champ
            break

    where_filter = {"champion": matched_champion} if matched_champion else None
    results = collection.query(query_texts=[query], n_results=n_results, where=where_filter)
    return results, matched_champion

for query in test_queries:
    results, champ = smart_search(query)
    print(f"\n=== 검색: '{query}' (감지된 챔피언: {champ}) ===")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"[{meta['champion']} | type: {meta.get('type', 'patch')}]")
        print(doc[:120], "...")