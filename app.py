# app.py  (검색 v2 데모)
# 실행: streamlit run app.py   (먼저 src/rag/build_index.py로 'gamehikking_v2' 컬렉션을 만들 것)
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rag"))

st.set_page_config(page_title="GameHikking AI Assistant", page_icon="🎮", layout="centered")


@st.cache_resource
def load_search():
    import search
    search.collection()               # 임베딩 모델·DB를 한 번만 로드
    return search


try:
    S = load_search()
except Exception as e:
    st.error(f"❌ 검색 v2를 불러오지 못했습니다. src/rag/build_index.py를 먼저 실행해 주세요.\n\n{e}")
    st.stop()

st.title("🎮 GameHikking LoL AI Assistant")
st.markdown("공식 패치노트(13.1~최신)와 Data Dragon 챔피언 정보에서 **근거를 찾아 원문 그대로** 답합니다.")

st.sidebar.header("⚙️ 검색 설정")
mode = st.sidebar.selectbox("게임 모드", ["소환사의 협곡", "칼바람", "아레나"])
show_debug = st.sidebar.checkbox("검색 과정 보기", value=True)

EXAMPLES = ["상대 다리우스, 나 나서스 탑 템트리 추천해줘", "아트록스 지옥사슬 쿨 줄었어?", "잭스 13.1 패치 왜 바뀐 거야?",
            "마이 궁 뭐야?", "몰왕 조합법 알려줘", "탱커 챔피언 추천해줘"]
st.caption("예시 질문")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state["q"] = ex

query = st.text_input("궁금한 챔피언·스킬·패치 내용을 입력하세요", key="q",
                      placeholder="예: 나서스 Q 스킬 알려줘")

if query.strip():
    with st.spinner("근거 검색 중..."):
        r = S.answer(query, mode=mode)

    st.subheader("💡 답변")
    c1, c2 = st.columns([1, 9])
    if r["icon"]:
        c1.image(r["icon"], width=56)
    c2.markdown(r["answer"])
    if r.get("item") and r["item"]["components"]:
        st.caption("조합 재료")
        comp = r["item"]["components"]
        cc = st.columns(len(comp) * 2 + 1)
        for i, c in enumerate(comp):
            box = cc[i * 2]
            if c["icon"]:
                box.image(c["icon"], width=48)
            box.caption(f"{c['name']}\n{c['gold']}G")
            cc[i * 2 + 1].markdown("### +" if i < len(comp) - 1 else "### →")
        if r["item"]["icon"]:
            cc[-1].image(r["item"]["icon"], width=56)
            cc[-1].caption(f"{r['item']['name']}\n{r['item']['gold']}G")
    if r.get("build"):
        order = r["build"]["order"]
        st.caption("구매 순서")
        cc = st.columns(len(order))
        for k, (box, c) in enumerate(zip(cc, order), 1):
            if c["icon"]:
                box.image(c["icon"], width=48)
            box.caption(f"{k}. {c['name']}\n{c['gold']}G")
    for url in r["sources"]:
        st.markdown(f"📎 출처: [{url}]({url})")

    if show_debug:
        P = r["parsed"]
        with st.expander("🔎 질문 해석"):
            st.json({"의도": P.intent, "챔피언": P.champion, "아이템": P.item,
                     "패치": P.patch and S.patch_label(P.patch), "패치 자동(최신 변경)": P.patch_auto,
                     "슬롯": P.slot, "역할군": P.role, "변경 이유 질문": P.why, "키워드": P.keywords,
                     "완화한 조건": r["relaxed"]})
        with st.expander(f"📦 검색된 근거 {len(r['hits'])}개 (★ = 선택된 근거)"):
            for i, h in enumerate(r["hits"][:15]):
                m = h["meta"]
                star = "★ " if i == r["best"] else ""
                st.markdown(f"**{star}[{i + 1}] {m.get('type')}** · {m.get('champion', m.get('name', ''))} "
                            f"· {m.get('patch', '')} · {m.get('slot', '')} · 거리 {h['dist']}")
                st.text(h["text"][:300])