# vectorDB_v2.py (모드 및 아이템/룬/주문 확장 데이터 포함 Vector DB 재구축)
import json
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

MODEL = "distiluse-base-multilingual-cased-v1"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)
client = chromadb.PersistentClient(path="../../chroma_db")

# 기존 컬렉션 삭제 후 재생성 (클린 적재)
try:
    client.delete_collection("gamehikking")
except Exception:
    pass

collection = client.get_or_create_collection(name="gamehikking", embedding_function=ef)

total_count = 0

# --- (1) 패치 청크 적재 (기본 모드: 소환사의 협곡) ---
header_re = re.compile(r"^(.+?) \| patch_(.+?) ===\n(.+)$", re.DOTALL)
chunks_path = Path("../../data/raw/chunks.txt")
if chunks_path.exists():
    patch_chunks = []
    for block in chunks_path.read_text(encoding="utf-8").strip().split("=== ")[1:]:
        m = header_re.match(block)
        if m:
            patch_chunks.append({
                "champion": m.group(1).strip(),
                "patch": m.group(2).strip(),
                "text": m.group(3).strip()
            })

    for i, c in enumerate(patch_chunks):
        collection.add(
            documents=[c["text"]],
            metadatas=[{
                "champion": c["champion"],
                "patch": c["patch"],
                "mode": "소환사의 협곡",
                "type": "patch_note"
            }],
            ids=[f"patch_{i}"]
        )
    print(f"✅ 패치 청크 {len(patch_chunks)}개 적재 완료")
    total_count += len(patch_chunks)

# --- (2) 챔피언 기본정보 추가 (모드: 공통) ---
base_path = Path("../../data/raw/champion_base_data.txt")
if base_path.exists():
    base_blocks = base_path.read_text(encoding="utf-8").strip().split("=== ")[1:]
    for i, block in enumerate(base_blocks):
        champion, text = block.split(" ===\n", 1)
        text = re.sub(r"<br\s*/?>", " ", text.strip())
        collection.add(
            documents=[text],
            metadatas=[{
                "champion": champion.strip(),
                "type": "base_info",
                "patch": "N/A",
                "mode": "공통"
            }],
            ids=[f"base_{i}"]
        )
    print(f"✅ 챔피언 기본정보 {len(base_blocks)}개 적재 완료")
    total_count += len(base_blocks)

# --- (3) 확장 데이터(아이템, 룬, 주문, 팁 등) 추가 (모드 메타데이터 포함) ---
extra_path = Path("../../data/chunks/ddragon_extra.jsonl")
if extra_path.exists():
    extra_records = []
    with open(extra_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                extra_records.append(json.loads(line))

    valid_extras = 0
    for i, r in enumerate(extra_records):
        meta = {
            "mode": r.get("mode", "공통"),
            "type": r.get("type", "unknown"),
            "name": r.get("name", "N/A"),
            "patch": "N/A"
        }
        # Chroma 메타데이터는 None 값을 허용하지 않으므로 icon이 있는 경우에만 추가
        if r.get("icon"):
            meta["icon"] = r["icon"]

        collection.add(
            documents=[r["text"]],
            metadatas=[meta],
            ids=[f"extra_{i}"]
        )
        valid_extras += 1

    print(f"✅ 확장 데이터(아이템/룬/주문 등) {valid_extras}개 적재 완료")
    total_count += valid_extras

print(f"\n🎉 Vector DB 재구축 최종 완료! 총 청크 수: {collection.count()}")

# --- (4) 모드 필터링이 포함된 스마트 검색 함수 예시 ---
known_champions = sorted({m["champion"] for m in collection.get()["metadatas"] if m.get("champion")}, key=len,
                         reverse=True)


def smart_search_with_mode(query, target_mode="소환사의 협곡", n_results=3):
    champ = next((c for c in known_champions if c in query), None)
    pm = re.search(r"(\d{2}-\d{2})", query)
    patch = pm.group(1) if pm else None

    # 모드 필터: 선택한 모드 또는 "공통" 데이터를 모두 포함
    mode_filter = {"mode": {"$in": [target_mode, "공통"]}}

    where = mode_filter
    if champ and patch:
        where = {"$and": [{"champion": champ}, {"patch": patch}, mode_filter]}
    elif champ:
        where = {"$and": [{"champion": champ}, mode_filter]}

    return collection.query(query_texts=[query], n_results=n_results, where=where), champ, patch, target_mode