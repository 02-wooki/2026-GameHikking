# app.py  (검색 v2 데모)
# 실행: streamlit run app.py   (먼저 src/rag/build_index.py로 'gamehikking_v2' 컬렉션을 만들 것)
# 라이트/다크: .streamlit/config.toml의 [theme.light]/[theme.dark] + 아래 TOKENS
import html
import re
import sys
from urllib.parse import urlparse
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "gpt"))

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
[class*="st-key-answer"] { position: relative; padding: 1.2rem 1.3rem 1.1rem;
  background: var(--paper); border: 1px solid var(--line); border-radius: 10px; }
.gh-head { display: flex; align-items: center; gap: .85rem; margin-bottom: .75rem; }
[class*="st-key-scroll_"], [data-testid="stLayoutWrapper"]:has(> [class*="st-key-scroll_"]) { display: none; }   /* 스크롤용 스크립트 자리는 화면에 공간을 차지하지 않게 */
.gh-head img { width: 52px; height: 52px; border-radius: 8px; border: 1px solid var(--line); }
.gh-q { font-size: .85rem; color: var(--muted); }
[class*="st-key-answer"] .stMarkdown p, [class*="st-key-answer"] .stMarkdown li { line-height: 1.8; }
[class*="st-key-answer"] .stMarkdown em { font-style: normal; color: var(--muted); font-size: .86rem; }
[class*="st-key-answer"] blockquote { margin: .6rem 0; padding: .7rem 1rem; background: var(--quote);
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
/* 입력창 아래 옵션 줄: 입력창보다 한 단계 작게 */
.st-key-opts { padding: .35rem .25rem 0; }
.st-key-opts label p, .st-key-opts .stCaption p { font-size: .8rem; color: var(--muted); }

/* 대화를 시작하면 제목을 작게 */
.stMarkdown h1.gh-title.compact { font-size: 1.6rem !important; margin-bottom: .6rem; }
.st-key-topbar { align-items: center; }

/* 채팅: 내 질문은 오른쪽 말풍선, 답변은 아바타 없이 카드만 */
div[data-testid="stChatMessage"] { background: transparent; padding: .25rem 0; gap: 0; }
[data-testid^="stChatMessageAvatar"] { display: none; }   /* 사람·로봇 아바타는 쓰지 않는다 */
/* 답변을 찾는 동안: 로딩 원과 문구 사이를 띄우고, 답변 카드 글자 위치에 맞춘다 */
div[data-testid="stChatMessage"] [data-testid="stSpinner"] { padding: .9rem 1.3rem; }
div[data-testid="stChatMessage"] [data-testid="stSpinner"] > div { gap: .75rem; align-items: center; }
div[data-testid="stChatMessage"] [data-testid="stSpinner"] p { color: var(--muted); font-size: .92rem; }
.gh-me { background: var(--accent); color: var(--paper); padding: .55rem .95rem; border-radius: 14px 14px 4px 14px;
  width: fit-content; max-width: min(34em, 85%); line-height: 1.6; margin-left: auto; }
.gh-mode { font-size: .72rem; opacity: .75; margin-top: .15rem; text-align: right; }
/* 빈 입력창이 웹폰트 로딩 전 높이로 크게 잡히는 문제: 내용 높이에 맞추고 최대 6줄 */
div[data-testid="stChatInput"] textarea { font-size: 1rem; height: auto !important; min-height: 1.6em;
  max-height: 9em; field-sizing: content; }
div[data-testid="stChatInput"] textarea::placeholder { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* 새 질문이 슥 올라오고, 답변이 살짝 뒤따라 올라온다.
   Streamlit은 재실행 때 기존 요소를 다시 만들지 않으므로 이미 있던 말풍선·카드는 다시 움직이지 않는다. */
@keyframes gh-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.gh-me { animation: gh-rise .34s cubic-bezier(.2, .7, .3, 1) both; }
[class*="st-key-answer"] { animation: gh-rise .38s cubic-bezier(.2, .7, .3, 1) .2s both; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
st.markdown(CSS.replace("__VARS__", VARS), unsafe_allow_html=True)
e = html.escape


RAG_DIR = Path(__file__).resolve().parent / "src" / "rag"
GPT_DIR = Path(__file__).resolve().parent / "src" / "gpt"


def _mtime(*files):
    return tuple(f.stat().st_mtime for f in files)


# 인자로 파일 수정 시각을 넘겨, 서버를 켜 둔 채 코드가 바뀌어도 예전 모듈을 계속 쓰지 않게 한다
@st.cache_resource
def load_search(ver):
    import importlib
    import search
    importlib.reload(search)
    search.collection()               # 임베딩 모델·DB를 한 번만 로드
    return search


@st.cache_resource
def load_minigpt(ver):
    import importlib
    import minigpt_selector as M
    importlib.reload(M)
    M._load()                         # 모델을 한 번만 로드 (체크포인트가 없으면 여기서 예외)
    return M


try:
    S = load_search(_mtime(RAG_DIR / "search.py", RAG_DIR / "recommend.py"))
except Exception as ex:
    st.error(f"검색 인덱스를 불러오지 못했습니다. src/rag/build_index.py를 먼저 실행한 뒤 새로고침하세요.\n\n{ex}")
    st.stop()

EXAMPLES = ["상대 다리우스, 나 나서스 탑 템트리 추천해줘", "아트록스 지옥사슬 쿨 줄었어?", "잭스 13.1 패치 왜 바뀐 거야?",
            "마이 궁 뭐야?", "몰왕 조합법 알려줘", "탱커 챔피언 추천해줘"]
MAX_TURNS = 30                        # 화면에 남겨 둘 최근 대화 수

if "turns" not in st.session_state:
    st.session_state.turns = []       # [{"q", "mode", "r", "mg", "mg_note"}]
turns = st.session_state.turns


def ask_example(x):
    st.session_state.pending = x


def clear_chat():
    st.session_state.turns = []


def item_html(c, k=None, final=False):
    img = f'<img src="{e(c["icon"])}" alt="">' if c.get("icon") else '<div style="height:46px"></div>'
    badge = f'<span class="k">{k}</span>' if k else ""
    gold = f'<div class="g">{c["gold"]}G</div>' if c.get("gold") is not None else ""
    return (f'<div class="gh-item{" final" if final else ""}"><div class="ic">{badge}{img}</div>'
            f'<div class="n">{e(str(c["name"]))}</div>{gold}</div>')


def source_label(url):
    """패치노트 URL을 '26.19 패치노트'처럼 읽기 쉬운 이름으로. 모르는 형식이면 도메인."""
    m = re.search(r"patch-(\d+)-(\d+)-notes", url)
    return f"{m.group(1)}.{m.group(2)} 패치노트" if m else urlparse(url).netloc.removeprefix("www.")


def run_query(q, mode):
    """검색 + (켜져 있으면) MiniGPT 비교 정보까지 계산해 한 턴으로 묶는다."""
    r = S.answer(q, mode=mode, selector=M.select if M else None)
    mg = mg_note = None
    if M and M.applies(r["parsed"], r["hits"]):
        mg = M.hit_scores(r["parsed"], r["hits"])
        mg_note = f"MiniGPT가 {r['best'] + 1}번 줄을 골랐습니다 (일치 확률 {mg[r['best']]:.2f})."
    elif M:
        mg_note = "이 질문은 MiniGPT 적용 대상이 아니라 규칙 기반으로 골랐습니다 (후보 1개 또는 학습하지 않은 질문 종류)."
    return {"q": q, "mode": mode, "r": r, "mg": mg, "mg_note": mg_note}


# 스크롤 애니메이터: 한 번만 설치되는 rAF 루프 하나가 스크롤 위치를 전부 맡는다.
# 목표는 '맨 아래'가 아니라 '새 질문 말풍선이 화면 위쪽에 오는 위치'. 답변은 그 아래에 나타나므로
# 처음부터 읽을 수 있고, 말풍선에 도착한 뒤에는 더 내려가지 않는다.
# 임계 감쇠 스프링이라 출발·도착이 부드럽고, 답변이 붙어 내려갈 공간이 생겨도 처음부터 다시 시작하지 않고 이어 간다.
FOLLOW_JS = """
(() => {
  if (window.__ghFollow) return;
  const box = () => document.querySelector('[data-testid="stAppScrollToBottomContainer"]');
  const TOP_GAP = 76;                 // 말풍선 위 여백 (상단 Deploy 줄에 가리지 않게)
  const W = 10;                       // 스프링 강도(rad/s). 클수록 빨리 도착
  let raf = 0, x = 0, v = 0, last = 0, until = 0, el = null, phase = 'idle', calm = 0;
  const stop = (ev) => { if (ev && !ev.isTrusted) return; cancelAnimationFrame(raf); raf = 0; phase = 'idle'; };
  // Streamlit은 맨 아래에 닿으면 '바닥 붙기' 모드가 되어 새 내용을 따라 내려간다.
  // 휠 이벤트를 받으면 그 모드를 끄므로, 스크롤하는 동안엔 가짜 휠 이벤트로 꺼 둔다.
  const unstick = () => el.dispatchEvent(new WheelEvent('wheel', { deltaY: -1 }));
  const targetOf = () => {
    const me = [...document.querySelectorAll('.gh-me')].at(-1);
    const max = el.scrollHeight - el.clientHeight;
    if (!me) return max;
    const msg = me.closest('[data-testid="stChatMessage"]') || me;
    const top = el.scrollTop + msg.getBoundingClientRect().top - el.getBoundingClientRect().top - TOP_GAP;
    return Math.max(0, Math.min(top, max));
  };
  const step = (t) => {
    if (!el || !el.isConnected) return stop();
    const gap = t - last; last = t;
    unstick();
    if (phase === 'settle') {
      // 질문을 보낸 직후엔 Streamlit이 화면을 그리느라 프레임이 띄엄띄엄 온다.
      // 그 사이에 움직이면 덜컹거리므로, 프레임이 고르게 3번 연속 올 때까지(최대 0.6초) 기다렸다 출발한다.
      calm = gap < 24 ? calm + 1 : 0;
      if (calm >= 3 || t > until - 100) { phase = 'move'; x = el.scrollTop; v = 0; }
      raf = requestAnimationFrame(step); return;
    }
    const dt = Math.min(gap / 1000, 1 / 50);   // 프레임이 밀려도 한 번에 크게 뛰지 않게
    const target = targetOf();
    v += (W * W * (target - x) - 2 * W * v) * dt;
    x += v * dt;
    el.scrollTop = x;
    if (Math.abs(target - x) < 0.5 && Math.abs(v) < 4 && t > until) return stop();
    raf = requestAnimationFrame(step);
  };
  window.__ghFollow = (ms) => {
    const b = box(); if (!b) return;
    const now = performance.now();
    until = Math.max(until, now + ms);
    if (raf && el === b) return;      // 이미 진행 중이면 시간만 늘리고 이어서 간다
    if (el !== b) { b.addEventListener('wheel', stop, { passive: true }); b.addEventListener('pointerdown', stop, { passive: true }); }
    el = b;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { unstick(); el.scrollTop = targetOf(); return; }
    phase = 'settle'; calm = 0; last = now;
    until = Math.max(until, now + 700);
    raf = requestAnimationFrame(step);
  };
})();
"""


def scroll_to_question(tag, n):
    """새 질문을 보내면 어디를 보고 있든 그 질문 말풍선이 화면 위쪽에 오도록 부드럽게 스크롤한다.
    tag="a"는 답변이 그려진 뒤 한 번 더 불러, 답변 덕에 내려갈 공간이 생긴 만큼 마저 이동하게 한다.
    n(턴 번호)을 스크립트에 넣어 매번 내용이 달라지게 해야 재실행 때 다시 실행된다."""
    # 대화 영역이 아니라 하단 고정 영역(st.bottom)에 둔다. 대화 사이에 끼워 넣으면 매 실행마다 요소 순서가 밀려,
    # 새 질문 말풍선이 잠깐 이전 답변 자리(펼치기 메뉴 포함)를 물려받아 보이는 문제가 생긴다.
    with st.bottom, st.container(key=f"scroll_{tag}"):
        st.html(f"<script>/* turn {n} {tag} */{FOLLOW_JS} window.__ghFollow(700);</script>",
                unsafe_allow_javascript=True)


def render_question(t):
    with st.chat_message("user"):
        note = f'<div class="gh-mode">{e(t["mode"])}</div>' if t["mode"] != "소환사의 협곡" else ""
        st.markdown(f'<div class="gh-me">{e(t["q"])}{note}</div>', unsafe_allow_html=True)


def render_answer(t, i):
    r = t["r"]
    with st.container(key=f"answer{i}"):
        if r["icon"]:
            st.markdown(f'<div class="gh-head"><img src="{e(r["icon"])}" alt=""></div>', unsafe_allow_html=True)
        st.markdown(r["answer"])

        if r.get("item") and r["item"]["components"]:
            comp = r["item"]["components"]
            parts = []
            for j, c in enumerate(comp):
                parts += [item_html(c), f'<div class="gh-op">{"+" if j < len(comp) - 1 else "="}</div>']
            parts.append(item_html(r["item"], final=True))
            st.markdown(f'<div class="gh-sub">조합 재료</div><div class="gh-row">{"".join(parts)}</div>',
                        unsafe_allow_html=True)

        if r.get("build"):
            row = "".join(item_html(c, k) for k, c in enumerate(r["build"]["order"], 1))
            st.markdown(f'<div class="gh-sub">구매 순서</div><div class="gh-row">{row}</div>', unsafe_allow_html=True)

        if r["sources"]:
            # st.markdown은 링크 안의 URL 글자를 다시 자동 링크로 바꿔 도장이 깨지므로 st.html로 그린다
            st.html("".join(f'<a class="gh-stamp" href="{e(u)}" target="_blank" title="{e(u)}">'
                            f'<b>출처</b>{e(source_label(u))}</a>' for u in r["sources"]))

    if show_debug:
        P = r["parsed"]
        with st.expander("질문을 이렇게 해석했어요"):
            st.json({"의도": P.intent, "챔피언": P.champion, "아이템": P.item,
                     "패치": P.patch and S.patch_label(P.patch), "패치 자동(최신 변경)": P.patch_auto,
                     "슬롯": P.slot, "역할군": P.role, "변경 이유 질문": P.why, "키워드": P.keywords,
                     "완화한 조건": r["relaxed"]})
        if t["mg_note"]:
            st.caption(t["mg_note"])
        mg = t["mg"]
        with st.expander(f"검색된 근거 {len(r['hits'])}개"):
            rows = []
            for j, h in enumerate(r["hits"][:15]):
                m = h["meta"]
                best = j == r["best"]
                tags = "".join(f'<span class="gh-tag">{e(str(x))}</span>' for x in (m.get("patch"), m.get("slot")) if x)
                rows.append(f'<div class="gh-hit{" best" if best else ""}"><div class="m">'
                            f'<b>{"선택됨 " if best else ""}{j + 1}. {e(str(m.get("type")))}, '
                            f'{e(str(m.get("champion", m.get("name", ""))))}</b>{tags} 거리 {h["dist"]}'
                            f'{f" · MiniGPT {mg[j]:.2f}" if mg else ""}</div>'
                            f'<div class="t">{e(h["text"][:300])}</div></div>')
            st.markdown("".join(rows), unsafe_allow_html=True)


# 입력창은 어디서 호출해도 화면 맨 아래에 고정된다. 먼저 읽어서, 첫 질문부터 바로 대화 화면으로 그린다.
typed = st.chat_input("챔피언, 스킬, 아이템, 패치를 물어보세요")

# 입력창 바로 아래 옵션 줄 (st.bottom: 입력창과 함께 화면 아래에 고정되는 영역)
with st.bottom, st.container(key="opts", horizontal=True, gap="medium", vertical_alignment="center"):
    show_debug = st.toggle("검색 근거 보기", value=True, key="opt_debug")
    use_minigpt = st.toggle("MiniGPT로 근거 고르기 (실험)", value=False, key="opt_minigpt",
                            help="검색된 후보 줄 중 답이 되는 줄을 규칙 대신 직접 학습한 MiniGPT가 고릅니다. "
                                 "스킬·패치 변경·기본 능력치 질문에서 후보가 2개 이상일 때만 적용됩니다.")
    M = None
    if use_minigpt:
        try:
            M = load_minigpt(_mtime(GPT_DIR / "minigpt_selector.py", GPT_DIR / "mini_gpt.py"))
        except Exception as ex:
            st.caption(":orange[MiniGPT를 불러오지 못해 규칙 기반으로 고릅니다]", help=str(ex))
q = (typed or st.session_state.pop("pending", None) or "").strip()
chatting = bool(turns or q)

# ---------- 상단 ----------
if chatting:
    with st.container(key="topbar", horizontal=True):
        st.markdown('<h1 class="gh-title compact">패치노트에 물어보세요</h1>', unsafe_allow_html=True)
        st.button("새 대화", on_click=clear_chat, type="tertiary", icon=":material/add_comment:")
else:
    st.markdown('<h1 class="gh-title">패치노트에<br>물어보세요</h1>'
                '<p class="gh-lede">챔피언, 스킬, 아이템, 패치 변경점을 물으면 공식 패치노트와 Data Dragon에서 찾은 문장을 '
                '고쳐 쓰지 않고 그대로, 출처와 함께 보여줍니다.</p>', unsafe_allow_html=True)

mode = st.segmented_control("게임 모드", ["소환사의 협곡", "칼바람", "아레나"], default="소환사의 협곡",
                            key="mode", label_visibility="collapsed") or "소환사의 협곡"

if not chatting:
    st.markdown('<div class="gh-hint">이런 질문을 해볼 수 있어요</div>', unsafe_allow_html=True)
    with st.container(key="examples", horizontal=True, gap="small"):
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=ask_example, args=(ex,), width="content")

# ---------- 대화 ----------
for i, t in enumerate(turns):
    render_question(t)
    with st.chat_message("assistant"):
        render_answer(t, i)

if q:
    t = {"q": q, "mode": mode}
    render_question(t)
    scroll_to_question("q", len(turns))
    with st.chat_message("assistant"):
        with st.spinner("패치노트에서 근거를 찾는 중..."):
            t = run_query(q, mode)
        render_answer(t, len(turns))
    scroll_to_question("a", len(turns))
    turns.append(t)
    del turns[:-MAX_TURNS]
