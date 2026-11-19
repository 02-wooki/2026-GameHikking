# src/rag/rag_pipeline.py
import chromadb
from pathlib import Path

# 1. 프로젝트 루트 경로를 기준으로 chroma_db 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
db_path = str(ROOT_DIR / "chroma_db")

print(f"📂 Chroma DB 경로: {db_path}")
client = chromadb.PersistentClient(path=db_path)

collection_name = "gamehikking"
try:
    collection = client.get_collection(name=collection_name)
    print(f"✅ Chroma DB 로드 성공! (총 문서 수: {collection.count()})")
except Exception as e:
    print(f"❌ 컬렉션을 찾을 수 없습니다. 에러: {e}")
    exit()


# 2. 정밀 RAG 검색 함수
def search_rag(query: str, n_results: int = 5):
    """
    사용자 질문을 기반으로 Vector DB에서 가장 관련성 높은 청크를 검색합니다.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    retrieved_chunks = results['documents'][0]
    metadatas = results['metadatas'][0]

    return retrieved_chunks, metadatas


# 3. RAG 기반 응답 생성 함수
def generate_rag_response(query: str):
    print(f"\n🔍 사용자 질문: '{query}'")

    # 관련 문서 검색 (상위 5개)
    chunks, metas = search_rag(query, n_results=5)

    print("\n📦 [Vector DB에서 검색된 상위 근거 자료]:")
    for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
        mode = meta.get('mode', '기본')
        print(f"--- [근거 {i}] (모드: {mode}) ---")
        print(chunk[:150].replace('\n', ' ') + "...")

    # 💡 [RAG 기반 최종 응답 생성 로직]:
    print("\n💡 [RAG 기반 최종 응답 생성]:")

    # 질의어에서 불용어 제거 후 핵심 타겟 키워드 추출
    target_keywords = [w for w in query.split() if w not in ["정보", "알려줘", "스킬", "어때", "뭐야", "소개해줘"]]
    if not target_keywords:
        target_keywords = query.split()

    best_chunk = None

    # 1순위: 핵심 키워드(예: '나서스', 'Q' 등)가 모두 포함된 청크 탐색
    for chunk, meta in zip(chunks, metas):
        if all(kw in chunk for kw in target_keywords):
            best_chunk = chunk
            break

    # 2순위: 핵심 키워드 중 하나라도 포함된 청크 탐색
    if not best_chunk:
        for chunk, meta in zip(chunks, metas):
            if any(kw in chunk for kw in target_keywords):
                best_chunk = chunk
                break

    # 3순위: 그래도 없으면 가장 유사도 높은 첫 번째 청크 사용
    if not best_chunk and chunks:
        best_chunk = chunks[0]

    # 최종 결과 출력
    if best_chunk:
        print(f"참고 문서 기반 답변:\n{best_chunk}")
    else:
        print("관련된 게임 정보를 찾지 못했습니다.")


if __name__ == "__main__":
    # 테스트 쿼리 실행
    generate_rag_response("나서스 Q 스킬 정보 알려줘")