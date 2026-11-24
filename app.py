# app.py  (검색 v2 데모)
# 실행: streamlit run app.py   (먼저 src/rag/build_index.py로 'gamehikking_v2' 컬렉션을 만들 것)
# 라이트/다크: .streamlit/config.toml의 [theme.light]/[theme.dark] + 아래 TOKENS
import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rag"))

st.set_page_config(page_title="GameHikking", page_icon="📜", layout="centered")

# 테마별 색 토큰 (라이트, 다크). Streamlit이 .stApp에 color-scheme을 걸어주므로
# CSS light-dark()가 ⋮ > Settings 테마 전환을 재실행 없이 바로 따라간다.
TOKENS = {
    "ink": ("#18222F", "#DCE3EA"), "muted": ("#5B6877", "#8C99A8"), "line": ("#C9D3DC", "#2E3A48"),
    "paper": ("#FFFFFF", "#1C2531"), "quote": ("#EEF2FF", "#1E2846"), "accent": ("#2546D9", "#8EA6FF"),
    "gold": ("#A8640F", "#E3A54A"), "chip": ("#FFFFFF", "#18212C"),
}
VARS = " ".join(f"--{k}: light-dark({lt}, {dk});" for k, (lt, dk) in TOKENS.items())

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap');
.stApp { __VARS__ }
.block-container { padding-top: 3rem; max-width: 720px; }
html, body, .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans KR', sans-serif; }

/* 제목: 이 페이지에서 유일하게 큰 목소리를 내는 요소 */
.stMarkdown h1.gh-title { font-family: 'Black Han Sans', sans-serif !important; font-weight: 400 !important;
  font-size: clamp(2.6rem, 8vw, 4.4rem) !important; line-height: 1.02; letter-spacing: -.01em;
  color: var(--ink); margin: 0 0 .9rem; padding: 0; }
.stMarkdown h1.gh-title a { display: none; }
.gh-lede { color: var(--muted); font-size: 1rem; line-height: 1.7; max-width: 34em; margin-bottom: 1.8rem; }

/* 검색창 */
div[data-testid="stTextInput"] input { font-size: 1.05rem; padding: .9rem 1rem; }
div[data-testid="stTextInput"] > div { border-radius: 8px; border-width: 1.5px; }

/* 예시 질문: 문장 그대로 보여주는 작은 링크형 버튼 */
.st-key-examples button { background: var(--chip); border: 1px solid var(--line); border-radius: 6px;
  min-height: 0; padding: .3rem .7rem; color: var(--ink); }
.st-key-examples button p { font-size: .85rem; }
.st-key-examples button:hover { border-color: var(--accent); color: var(--accent); }
.st-key-examples button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.gh-hint { font-size: .85rem; color: var(--muted); margin: 1.4rem 0 .5rem; }

/* 답변: 패치노트 원문을 인용 블록처럼 */
.st-key-answer { position: relative; margin-top: 2.2rem; padding: 1.3rem 1.4rem 1.2rem;
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px; }
.gh-head { display: flex; align-items: center; gap: .85rem; }
.gh-head img { width: 52px; height: 52px; border-radius: 8px; border: 1px solid var(--line); }
.gh-q { font-size: .85rem; color: var(--muted); }
.st-key-answer .stMarkdown p, .st-key-answer .stMarkdown li { line-height: 1.8; }
.st-key-answer .stMarkdown em { font-style: normal; color: var(--muted); font-size: .86rem; }
.st-key-answer blockquote { margin: .6rem 0; padding: .7rem 1rem; background: var(--quote);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; }

/* 출처 도장 */
.gh-stamp { display: inline-flex; gap: .5rem; align-items: baseline; margin-top: 1rem; padding: .4rem .75rem;
  border: 1.5px dashed var(--accent); border-radius: 6px; font-size: .82rem; color: var(--accent) !important;
  text-decoration: none !important; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gh-stamp b { font-weight: 700; }
.gh-stamp:hover { background: var(--quote); }

/* 아이템 줄 (조합·구매 순서) */
.gh-sub { font-size: .88rem; font-weight: 700; color: var(--ink); margin: 1.2rem 0 .6rem; }
.gh-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: .7rem .5rem; }
.gh-item { width: 70px; text-align: center; }
.gh-item .ic { position: relative; width: 46px; margin: 0 auto; }
.gh-item img { width: 46px; height: 46px; border-radius: 6px; border: 1px solid var(--line); display: block; }
.gh-item.final img { border: 2px solid var(--gold); }
.gh-item .n { font-size: .72rem; line-height: 1.3; margin-top: .3rem; word-break: keep-all; color: var(--ink); }
.gh-item .g { font-size: .72rem; font-weight: 700; color: var(--gold); }
.gh-item .k { position: absolute; left: -7px; top: -7px; min-width: 19px; height: 19px; border-radius: 50%;
  background: var(--accent); color: var(--paper); font-size: .68rem; font-weight: 700; display: grid; place-items: center; }
.gh-op { font-size: 1.1rem; color: var(--muted); padding-top: .7rem; }

/* 검색 근거 */
.gh-hit { padding: .65rem 0; border-bottom: 1px solid var(--line); }
.gh-hit:last-child { border-bottom: 0; }
.gh-hit .m { font-size: .8rem; color: var(--muted); }
.gh-hit .m b { color: var(--ink); font-weight: 500; }
.gh-hit.best .m b { color: var(--accent); font-weight: 700; }
.gh-hit .t { font-size: .82rem; line-height: 1.6; margin-top: .2rem; color: var(--ink); opacity: .8; }
.gh-tag { display: inline-block; font-size: .7rem; padding: 0 .4rem; margin-left: .3rem; border: 1px solid var(--line); border-radius: 4px; }
.gh-empty { margin-top: 2.6rem; color: var(--muted); font-size: .9rem; }
.gh-side { font-size: .8rem; color: var(--muted); line-height: 1.7; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS.replace("__VARS__", VARS), unsafe_allow_html=True)
e = html.escape


@st.cache_resource
def load_search():
    import search
    search.collection()               # 임베딩 모델·DB를 한 번만 로드
    return search


try:
    S = load_search()
except Exception as ex:
    st.error(f"검색 인덱스를 불러오지 못했습니다. src/rag/build_index.py를 먼저 실행한 뒤 새로고침하세요.\n\n{ex}")
    st.stop()

with st.sidebar:
    show_debug = st.toggle("검색 과정 보기", value=True)
    st.markdown('<div class="gh-side">공식 패치노트 13.1부터 최신 패치까지, Data Dragon 챔피언·아이템 정보를 검색합니다.<br><br>'
                '화면 테마는 오른쪽 위 ⋮ 메뉴의 Settings에서 바꿀 수 있습니다.</div>', unsafe_allow_html=True)

st.markdown('<h1 class="gh-title">패치노트에<br>물어보세요</h1>'
            '<p class="gh-lede">챔피언, 스킬, 아이템, 패치 변경점을 물으면 공식 패치노트와 Data Dragon에서 찾은 문장을 '
            '고쳐 쓰지 않고 그대로, 출처와 함께 보여줍니다.</p>', unsafe_allow_html=True)

query = st.text_input("질문", key="q", label_visibility="collapsed", placeholder="예: 아트록스 W 쿨타임 줄었어?")
mode = st.segmented_control("게임 모드", ["소환사의 협곡", "칼바람", "아레나"], default="소환사의 협곡",
                            label_visibility="collapsed") or "소환사의 협곡"

EXAMPLES = ["상대 다리우스, 나 나서스 탑 템트리 추천해줘", "아트록스 지옥사슬 쿨 줄었어?", "잭스 13.1 패치 왜 바뀐 거야?",
            "마이 궁 뭐야?", "몰왕 조합법 알려줘", "탱커 챔피언 추천해줘"]


def set_q(x):
    st.session_state["q"] = x


def item_html(c, k=None, final=False):
    img = f'<img src="{e(c["icon"])}" alt="">' if c.get("icon") else '<div style="height:46px"></div>'
    badge = f'<span class="k">{k}</span>' if k else ""
    gold = f'<div class="g">{c["gold"]}G</div>' if c.get("gold") is not None else ""
    return (f'<div class="gh-item{" final" if final else ""}"><div class="ic">{badge}{img}</div>'
            f'<div class="n">{e(str(c["name"]))}</div>{gold}</div>')


if not query.strip():
    st.markdown('<div class="gh-hint">이런 질문을 해볼 수 있어요</div>', unsafe_allow_html=True)
    with st.container(key="examples", horizontal=True, gap="small"):
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=set_q, args=(ex,), width="content")
else:
    with st.spinner("패치노트에서 근거를 찾는 중..."):
        r = S.answer(query, mode=mode)

    with st.container(key="answer"):
        icon = f'<img src="{e(r["icon"])}" alt="">' if r["icon"] else ""
        st.markdown(f'<div class="gh-head">{icon}<div class="gh-q">“{e(query)}”에 대한 근거</div></div>',
                    unsafe_allow_html=True)
        st.markdown(r["answer"])

        if r.get("item") and r["item"]["components"]:
            comp = r["item"]["components"]
            parts = []
            for i, c in enumerate(comp):
                parts += [item_html(c), f'<div class="gh-op">{"+" if i < len(comp) - 1 else "="}</div>']
            parts.append(item_html(r["item"], final=True))
            st.markdown(f'<div class="gh-sub">조합 재료</div><div class="gh-row">{"".join(parts)}</div>',
                        unsafe_allow_html=True)

        if r.get("build"):
            row = "".join(item_html(c, k) for k, c in enumerate(r["build"]["order"], 1))
            st.markdown(f'<div class="gh-sub">구매 순서</div><div class="gh-row">{row}</div>', unsafe_allow_html=True)

        if r["sources"]:
            st.markdown("".join(f'<a class="gh-stamp" href="{e(u)}" target="_blank"><b>출처</b>{e(u)}</a>'
                                for u in r["sources"]), unsafe_allow_html=True)

    st.button("다른 질문 하기", on_click=set_q, args=("",), type="tertiary")

    if show_debug:
        P = r["parsed"]
        with st.expander("질문을 이렇게 해석했어요"):
            st.json({"의도": P.intent, "챔피언": P.champion, "아이템": P.item,
                     "패치": P.patch and S.patch_label(P.patch), "패치 자동(최신 변경)": P.patch_auto,
                     "슬롯": P.slot, "역할군": P.role, "변경 이유 질문": P.why, "키워드": P.keywords,
                     "완화한 조건": r["relaxed"]})
        with st.expander(f"검색된 근거 {len(r['hits'])}개"):
            rows = []
            for i, h in enumerate(r["hits"][:15]):
                m = h["meta"]
                best = i == r["best"]
                tags = "".join(f'<span class="gh-tag">{e(str(x))}</span>' for x in (m.get("patch"), m.get("slot")) if x)
                rows.append(f'<div class="gh-hit{" best" if best else ""}"><div class="m">'
                            f'<b>{"선택됨 " if best else ""}{i + 1}. {e(str(m.get("type")))}, '
                            f'{e(str(m.get("champion", m.get("name", ""))))}</b>{tags} 거리 {h["dist"]}</div>'
                            f'<div class="t">{e(h["text"][:300])}</div></div>')
            st.markdown("".join(rows), unsafe_allow_html=True)
