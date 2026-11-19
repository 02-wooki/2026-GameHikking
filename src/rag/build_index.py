# src/rag/build_index.py
"""검색 v2 Vector DB 구축: data/processed/*.jsonl -> Chroma 컬렉션 'gamehikking_v2'
기존 'gamehikking' 컬렉션(v1)은 건드리지 않는다 (비교 평가용).

문서 단위 (작게 쪼갬):
  patch_change  : 패치 변경 1줄            meta: champion, patch, patch_order, slot, skill, field, url, record, order
  patch_context : (패치, 챔피언) 개발자 코멘트  meta: champion, patch, patch_order, url, record
  skill         : 스킬 1개 (패시브/QWER)     meta: champion, slot, skill, icon
  stats         : 챔피언 기본 능력치          meta: champion
  profile       : 칭호·역할군                meta: champion, roles, role_<Tag>=True, icon
  tips          : 플레이 팁·상대 팁          meta: champion
  item          : items.jsonl의 아이템 x 모드 1개   meta: name, item_id, gold, icon, mode
  spell/rune    : ddragon_extra.jsonl (모드 태그 유지)
모든 문서에 mode 메타데이터 (패치 변경 = 소환사의 협곡, 챔피언 정보 = 공통)

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/rag/build_index.py
"""
import json
import time
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EXTRA = ROOT / "data" / "chunks" / "ddragon_extra.jsonl"
DB_PATH = ROOT / "chroma_db"
COLLECTION = "gamehikking_v2"
MODEL = "distiluse-base-multilingual-cased-v1"
SLOT_LABEL = {"패시브": "기본 지속 효과", "Q": "Q", "W": "W", "E": "E", "R": "R"}


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def patch_label(p):
    """'13-01' -> '13.1', '25-S1-1' -> '25.S1.1' (공식 표기)"""
    return ".".join(str(int(x)) if x.isdigit() else x for x in p.split("-"))


def stat_text(c):
    s = c["stats"]
    return (f"{c['name']} 기본 능력치 - 체력 {s['hp']} (레벨당 +{s['hpperlevel']}), "
            f"마나 {s['mp']} (레벨당 +{s['mpperlevel']}), 공격력 {s['attackdamage']} (레벨당 +{s['attackdamageperlevel']}), "
            f"공격 속도 {s['attackspeed']} (레벨당 +{s['attackspeedperlevel']}%), 방어력 {s['armor']} (레벨당 +{s['armorperlevel']}), "
            f"마법 저항력 {s['spellblock']} (레벨당 +{s['spellblockperlevel']}), 이동 속도 {s['movespeed']}, "
            f"사거리 {s['attackrange']}")


docs, metas, ids = [], [], []


def add(i, text, meta):
    ids.append(i)
    docs.append(text)
    metas.append({k: v for k, v in meta.items() if v is not None})   # Chroma는 None 금지


def main():
    patches = load_jsonl(PROC / "patch_notes.jsonl")
    champs = load_jsonl(PROC / "champions.jsonl")

    for r in patches:
        base = dict(mode="소환사의 협곡", champion=r["champion"], patch=r["patch"],
                    patch_order=r["patch_order"], url=r["url"], record=r["id"])
        head = f"{r['champion']} {patch_label(r['patch'])} 패치"
        for j, c in enumerate(r["changes"]):
            add(f"pc_{r['id']}_{j}", f"{head} - {c['text']}",
                dict(base, type="patch_change", slot=c["slot"], skill=c["skill"], field=c["field"], order=j))
        note = " ".join(x for x in (r["summary"], r["context"]) if x)
        if note:
            add(f"ctx_{r['id']}", f"{head} 변경 의도: {note}", dict(base, type="patch_context"))

    for c in champs:
        base = dict(mode="공통", champion=c["name"], version=c["version"])
        prof = dict(base, type="profile", roles=",".join(c["roles"]), icon=c["icon"])
        for t in c["tags"]:
            prof[f"role_{t}"] = True
        add(f"profile_{c['id']}", f"{c['name']} ({c['title']}) - 역할군: {', '.join(c['roles'])}. 자원: {c['partype']}.", prof)
        for s in c["skills"]:
            add(f"skill_{c['id']}_{s['slot']}", f"{c['name']} {SLOT_LABEL[s['slot']]} - {s['name']}: {s['description']}",
                dict(base, type="skill", slot=s["slot"], skill=s["name"], icon=s["icon"]))
        add(f"stats_{c['id']}", stat_text(c), dict(base, type="stats"))
        tips = " ".join(c["allytips"] + c["enemytips"])
        if tips:
            add(f"tips_{c['id']}", f"{c['name']} 플레이 팁: {' '.join(c['allytips'])} 상대 팁: {' '.join(c['enemytips'])}",
                dict(base, type="tips"))

    items_path = PROC / "items.jsonl"
    use_items = items_path.exists()
    if use_items:
        for it in load_jsonl(items_path):
            recipe = f" 조합: {' + '.join(x['name'] for x in it['from'])}." if it["from"] else ""
            text = (f"아이템 {it['name']}: {it['plaintext']} 능력치: {it['stats'] or '없음'}. "
                    f"효과: {it['effect'][:300]} 가격 {it['gold']['total']}골드.{recipe}")
            for mode in it["modes"]:
                add(f"item_{it['id']}_{mode}", text, dict(type="item", mode=mode, name=it["name"], item_id=it["id"],
                                                          gold=it["gold"]["total"], icon=it["icon"]))
    if EXTRA.exists():
        keep = ("spell", "rune") if use_items else ("item", "spell", "rune")
        for i, r in enumerate(load_jsonl(EXTRA)):
            if r.get("type") in keep:                         # 설정·팁은 champions.jsonl, 아이템은 items.jsonl로 대체
                add(f"extra_{i}", r["text"], dict(type=r["type"], mode=r.get("mode", "공통"),
                                                   name=r.get("name"), icon=r.get("icon")))

    print("문서 종류:", dict(Counter(m["type"] for m in metas)))
    print(f"총 {len(docs)}개 임베딩 시작 (모델: {MODEL})")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})
    t0 = time.time()
    for i in range(0, len(docs), 500):
        col.add(documents=docs[i:i + 500], metadatas=metas[i:i + 500], ids=ids[i:i + 500])
        print(f"  {min(i + 500, len(docs))}/{len(docs)} ({time.time() - t0:.0f}초)")
    print(f"\n완료: 컬렉션 '{COLLECTION}' 문서 {col.count()}개 | 경로 {DB_PATH}")
    print("기존 컬렉션 목록:", [getattr(c, "name", c) for c in client.list_collections()])


if __name__ == "__main__":
    main()