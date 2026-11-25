# admin.py  (질문 로그 검토 도구 — 로컬 전용, 공개 사이트에 올리지 않는다)
# 실행 (프로젝트 루트에서):
#   bash deploy/pull_logs.sh                      # 파이 로그를 logs/pi/ 로 받기 (선택)
#   /usr/local/bin/python3.10 -m streamlit run admin.py --server.port 8520 --server.address 127.0.0.1
# 검토 결과(정답 의도·확인함·메모)는 고르는 즉시 그 로그 파일(queries.jsonl)의 해당 줄에 써 넣는다
#   (intent_gold, reviewed, review_note 칸). 같은 내용을 logs/review_labels.jsonl 에도 덧붙여 두는데,
#   pull_logs.sh가 logs/pi/queries.jsonl 을 파이 원본으로 통째로 바꿔 놓아도 다음 화면에서 이 기록으로 다시 채우기 위해서다.
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "rag"))
import query_log as QL  # noqa: E402

st.set_page_config(page_title="GameHikking 로그 검토", page_icon="🗂️", layout="wide")

LABELS_PATH = ROOT / "logs" / "review_labels.jsonl"
SOURCES = {"파이 로그 (logs/pi)": ROOT / "logs" / "pi", "이 맥 (logs)": ROOT / "logs"}
PI_SOURCE = "파이 로그 (logs/pi)"
# 파이에서 로그를 찾을 위치 (앞쪽 우선): 새 위치(저장소 밖) -> 옮기기 전 위치(저장소 안)
PI_LOG_DIRS = ["/home/sysadmin/gamehikking-logs", "/home/sysadmin/2026-GameHikking/logs"]
# 실험 10(의도 분류)과 같은 9개 의도
INTENTS = {"overview": "챔피언 개요", "skill": "스킬 설명", "patch": "패치 변경", "why": "변경 이유",
           "stats": "능력치", "build": "템트리", "item": "아이템", "role": "역할군 추천", "other": "기타"}
REASONS = {"suspect": "자동 점검 suspect", "fail": "자동 점검 fail", "down": "👎 받음",
           "rephrased": "다시 물어봄", "missed": "못 알아들음", "noise": "noise 분류"}
PAGE = 15
e = html.escape

TOKENS = {"ink": ("#18222F", "#DCE3EA"), "muted": ("#5B6877", "#8C99A8"), "line": ("#C9D3DC", "#2E3A48"),
          "paper": ("#FFFFFF", "#1C2531"), "accent": ("#2546D9", "#8EA6FF"), "warn": ("#A8640F", "#E3A54A"),
          "bad": ("#B42318", "#F97066"), "chip": ("#EEF2FF", "#1E2846")}
VARS = " ".join(f"--{k}: light-dark({a}, {b});" for k, (a, b) in TOKENS.items())
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap');
.stApp { __VARS__ }
html, body, .stMarkdown, button, input, textarea { font-family: 'IBM Plex Sans KR', sans-serif; }
.block-container { padding-top: 2.4rem; max-width: 1100px; }
.ad-sub { color: var(--muted); font-size: .88rem; margin: -.4rem 0 1rem; }
.ad-q { font-size: 1.02rem; font-weight: 500; color: var(--ink); }
.ad-meta { font-size: .78rem; color: var(--muted); margin-top: .15rem; }
.ad-tag { display: inline-block; font-size: .72rem; padding: .05rem .45rem; border-radius: 999px; margin-right: .3rem;
  border: 1px solid var(--line); color: var(--muted); }
.ad-tag.fail, .ad-tag.down { color: var(--bad); border-color: var(--bad); }
.ad-tag.suspect, .ad-tag.rephrased { color: var(--warn); border-color: var(--warn); }
.ad-tag.missed, .ad-tag.noise { color: var(--accent); border-color: var(--accent); }
.ad-tag.done { background: var(--chip); color: var(--accent); border-color: transparent; }
.ad-kv { font-size: .82rem; line-height: 1.7; color: var(--ink); }
.ad-kv b { color: var(--muted); font-weight: 500; display: inline-block; min-width: 6.5em; }
.ad-why li { font-size: .82rem; }
.ad-ctx { font-size: .8rem; color: var(--muted); line-height: 1.7; }
.ad-ctx .me { color: var(--ink); font-weight: 700; }
[class*="st-key-card"] { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1.1rem; }
</style>""".replace("__VARS__", VARS), unsafe_allow_html=True)


# ---------- 데이터 ----------
def load_rows(log_dir):
    """query_log.load()를 그대로 쓰되 읽을 폴더만 바꾼다 (👍/👎·다시 물어봄 계산 포함)"""
    QL.LOG_PATH, QL.FEEDBACK_PATH = log_dir / "queries.jsonl", log_dir / "feedback.jsonl"
    return QL.load()


def load_labels():
    labels = {}
    if LABELS_PATH.exists():
        for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                labels[x["id"]] = x
    return labels


def save_label(log_id, **change):
    """기록 파일(review_labels.jsonl)에 현재 값 + 바뀐 칸을 한 줄 추가"""
    cur = {k: v for k, v in load_labels().get(log_id, {}).items() if k not in ("id", "ts")}
    cur.update(change)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": log_id, **cur, "ts": datetime.now(QL.KST).isoformat(timespec="seconds")},
                           ensure_ascii=False) + "\n")


LOG_FIELDS = {"intent_gold": "intent_gold", "reviewed": "reviewed", "note": "review_note"}   # 기록 칸 -> 로그 칸


def apply_labels(log_dir, labels, only=None):
    """검토 기록을 로그 파일의 해당 줄에 써 넣는다. 이미 같으면 파일을 건드리지 않음. 바뀐 칸 수를 돌려준다.
    파일 전체를 임시 파일에 쓴 뒤 바꿔치기(os.replace)해서, 쓰는 도중 끊겨도 로그가 반쯤 깨지지 않게 한다.
    로컬 앱이 같은 파일에 질문을 덧붙이는 중일 수 있어, 읽은 뒤 크기가 바뀌었으면 다시 읽어서 처리한다."""
    path = log_dir / "queries.jsonl"
    for _ in range(3):
        if not path.exists():
            return 0
        size = path.stat().st_size
        out, changed = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            lab = labels.get(x.get("id"))
            if lab and (only is None or x["id"] in only):
                diff = {f: lab[k] for k, f in LOG_FIELDS.items() if k in lab and x.get(f) != lab[k]}
                if diff:
                    x.update(diff)
                    changed += len(diff)
                    line = json.dumps(x, ensure_ascii=False)
            out.append(line)
        if not changed:
            return 0
        if path.stat().st_size != size:
            continue
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return changed
    return 0


def reasons_of(x):
    """이 질문을 검토 목록에 올린 이유들"""
    out = []
    if x["label"] == "noise":
        return ["noise"]
    st_ = x["check"]["status"]
    if st_ in ("suspect", "fail"):
        out.append(st_)
    if x.get("vote") == 0:
        out.append("down")
    if x.get("rephrased"):
        out.append("rephrased")
    if not x.get("understood"):
        out.append("missed")
    return out


def on_label(log_id, field, key, log_dir):
    """고르는 즉시: 기록 파일에 한 줄 + 로그 파일의 그 줄에 반영"""
    save_label(log_id, **{field: st.session_state[key]})
    apply_labels(log_dir, load_labels(), only={log_id})


def pull_pi_logs():
    """deploy/pull_logs.sh 실행 -> (성공 여부, 안내 문구). 파이가 꺼져 있거나 Wi-Fi가 끊겨도 멈추지 않게 시간 제한."""
    script = ROOT / "deploy" / "pull_logs.sh"
    if not script.exists():
        return False, "deploy/pull_logs.sh가 없습니다. 저장소를 최신으로 받아 주세요."
    pi = os.environ.get("PI", "pi-out")
    find = "for d in " + " ".join(PI_LOG_DIRS) + "; do if [ -f $d/queries.jsonl ]; then echo $d; break; fi; done; true"
    try:
        probe = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", pi, find],
                               capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        return False, "파이에 연결하지 못했습니다 (25초 초과). 파이 Wi-Fi가 끊겼을 수 있습니다. 잠시 뒤 다시 시도해 주세요."
    if probe.returncode != 0:
        why = (probe.stderr.strip().splitlines() or ["알 수 없는 오류"])[-1]
        return False, f"파이에 연결하지 못했습니다: {why}"
    log_dir = probe.stdout.strip()
    if not log_dir:
        return False, "파이에 아직 쌓인 로그가 없습니다."
    try:
        run = subprocess.run(["bash", str(script)], cwd=ROOT, capture_output=True, text=True, timeout=180,
                             env={**os.environ, "PI": pi, "PI_LOG_DIR": log_dir})
    except subprocess.TimeoutExpired:
        return False, "받는 중 시간이 초과됐습니다 (3분). 연결 상태를 확인하고 다시 시도해 주세요."
    if run.returncode != 0:
        why = (run.stderr.strip().splitlines() or run.stdout.strip().splitlines() or ["알 수 없는 오류"])[-1]
        return False, f"받기에 실패했습니다: {why}"
    q = ROOT / "logs" / "pi" / "queries.jsonl"
    n = sum(1 for line in q.read_text(encoding="utf-8").splitlines() if line.strip()) if q.exists() else 0
    when = datetime.now(QL.KST).strftime("%H:%M")
    return True, f"{when} 파이 로그를 받았습니다 · 질문 {n}줄 ({log_dir})"


# ---------- 상단 ----------
st.markdown("## 질문 로그 검토")
st.markdown('<div class="ad-sub">자동 점검에 걸렸거나 👎·재질문을 받은 질문을 확인하고, 실험 10에 쓸 정답 의도를 답니다. '
            '이 도구는 이 컴퓨터에서만 열립니다.</div>', unsafe_allow_html=True)

top = st.columns([3, 1], vertical_alignment="bottom")
# 버튼을 '로그 위치' 선택보다 먼저 만든다: 받은 직후 같은 실행 안에서 선택을 '파이 로그'로 바꿔야 하므로
if top[1].button("파이 로그 받아오기", icon=":material/cloud_download:", width="stretch", type="primary"):
    with st.spinner("파이에서 로그를 받는 중..."):
        ok, msg = pull_pi_logs()
    st.session_state.pull_msg = (ok, msg)
    if ok:
        st.session_state.src = PI_SOURCE

available = {k: v for k, v in SOURCES.items() if (v / "queries.jsonl").exists()}
if "pull_msg" in st.session_state:
    ok, msg = st.session_state.pull_msg
    (st.caption if ok else st.error)(msg)
if not available:
    st.info("읽을 로그가 없습니다. 오른쪽 위 '파이 로그 받아오기'를 누르거나, 로컬에서 앱을 써서 logs/에 질문을 쌓아 주세요.")
    st.stop()
if st.session_state.get("src") not in available:
    st.session_state.src = list(available)[0]
src_name = top[0].segmented_control("로그 위치", list(available), key="src") or list(available)[0]
log_dir = available[src_name]
apply_labels(log_dir, load_labels())        # 로그를 새로 받아 기록이 빠졌으면 다시 채움 (이미 같으면 그대로)
rows = load_rows(log_dir)

typed = [x for x in rows if x["source"] == "typed"]
questions = [x for x in typed if x["label"] == "question"]
needs = [x for x in typed if reasons_of(x) and x["label"] == "question"]
m = st.columns(6)
m[0].metric("직접 입력", len(typed), help=f"예시 버튼 {len(rows) - len(typed)}개는 제외")
m[1].metric("질문", len(questions), help=f"noise {len(typed) - len(questions)}개")
m[2].metric("검토 필요", len(needs))
m[3].metric("👍 / 👎", f"{sum(x.get('vote') == 1 for x in typed)} / {sum(x.get('vote') == 0 for x in typed)}")
m[4].metric("확인함", sum(1 for x in typed if x.get("reviewed")))
m[5].metric("정답 의도 달림", sum(1 for x in typed if x.get("intent_gold")))

# ---------- 필터 ----------
picked = st.pills("검토 이유", list(REASONS), format_func=REASONS.get, selection_mode="multi",
                  default=["suspect", "fail", "down", "rephrased", "missed"], key="reasons")
f = st.columns([2, 1, 1], vertical_alignment="bottom")
needle = f[0].text_input("질문 검색", placeholder="예: 아트록스", key="needle")
hide_done = f[1].toggle("확인한 것 숨기기", value=True, key="hide_done")
with_examples = f[2].toggle("예시 버튼 포함", value=False, key="with_ex")

pool = rows if with_examples else typed
shown = [x for x in pool if set(reasons_of(x)) & set(picked or [])]
if needle.strip():
    shown = [x for x in shown if needle.strip() in x["query"]]
shown.sort(key=lambda x: x["ts"], reverse=True)
todo = [x for x in shown if not x.get("reviewed")]
done = [x for x in shown if x.get("reviewed")]

by_session = {}
for x in rows:
    by_session.setdefault(x["session"], []).append(x)
for xs in by_session.values():
    xs.sort(key=lambda x: x["ts"])


def render_card(x):
    lid = x["id"]
    tags = "".join(f'<span class="ad-tag {r}">{REASONS[r]}</span>' for r in reasons_of(x))
    if x.get("reviewed"):
        tags += '<span class="ad-tag done">확인함</span>'
    if x["label"] == "noise":
        tags += f'<span class="ad-tag">{e(str(x.get("noise_type")))}</span>'
    when = x["ts"][5:16].replace("T", " ")
    with st.container(key=f"card_{lid}"):
        c = st.columns([3.2, 1.6])
        with c[0]:
            st.markdown(f'<div>{tags}</div><div class="ad-q">{e(x["query"])}</div>'
                        f'<div class="ad-meta">{when} · {e(x["mode"])} · 세션 {e(x["session"])} {x["turn"] + 1}번째'
                        f'{" · MiniGPT 켬" if x.get("minigpt") else ""}</div>', unsafe_allow_html=True)
            why = x["check"]["reasons"]
            if why:
                st.markdown('<ul class="ad-why">' + "".join(f"<li>{e(w)}</li>" for w in why) + "</ul>",
                            unsafe_allow_html=True)
            with st.expander("해석·답변·앞뒤 질문"):
                P = x.get("parsed") or {}
                a = x.get("answer") or {}
                kv = [("규칙 해석 의도", P.get("intent")), ("챔피언", P.get("champion")), ("슬롯", P.get("slot")),
                      ("패치", P.get("patch")), ("아이템", P.get("item")), ("역할군", P.get("role")),
                      ("이유 질문", P.get("why")), ("핵심어", ", ".join(P.get("keywords") or []) or None),
                      ("정규화", P.get("normalized")), ("근거 종류", a.get("type")),
                      ("완화한 조건", ", ".join(a.get("relaxed") or []) or None), ("답변 첫 줄", a.get("head"))]
                st.markdown('<div class="ad-kv">' + "".join(f"<div><b>{k}</b>{e(str(v))}</div>" for k, v in kv if v)
                            + "</div>", unsafe_allow_html=True)
                ctx = by_session.get(x["session"], [])
                i = next((k for k, y in enumerate(ctx) if y["id"] == lid), 0)
                near = ctx[max(0, i - 2): i + 3]
                if len(near) > 1:
                    st.markdown('<div class="ad-ctx">같은 세션 앞뒤 질문<br>' + "<br>".join(
                        f'<span class="{"me" if y["id"] == lid else ""}">{y["ts"][11:16]} {e(y["query"])}</span>'
                        for y in near) + "</div>", unsafe_allow_html=True)
        with c[1]:
            opts = list(INTENTS)
            guess = (x.get("parsed") or {}).get("intent")
            st.selectbox("정답 의도", opts, index=opts.index(x["intent_gold"]) if x.get("intent_gold") in opts else None,
                         placeholder="고르기", format_func=lambda k: f"{INTENTS[k]} ({k})",
                         key=f"gold_{lid}", on_change=on_label, args=(lid, "intent_gold", f"gold_{lid}", log_dir),
                         help=f"규칙 해석: {guess}" if guess else None)
            st.text_input("메모", value=x.get("review_note") or "", key=f"note_{lid}", placeholder="선택",
                          on_change=on_label, args=(lid, "note", f"note_{lid}", log_dir))
            st.toggle("확인함", value=bool(x.get("reviewed")), key=f"rev_{lid}",
                      on_change=on_label, args=(lid, "reviewed", f"rev_{lid}", log_dir))


def render_section(items, name, title=None):
    """카드 목록 + 쪽 나누기. title이 있으면 섹션 제목을 단다 ('확인 안 함' / '확인함')"""
    pages = max(1, -(-len(items) // PAGE))
    head = st.columns([4, 1], vertical_alignment="bottom")
    label = f"{title} · {len(items)}개" if title else f"{len(items)}개"
    head[0].markdown(f"#### {label}" if title else f"**{label}** · 최근 순"
                     + (f" · {pages}쪽" if pages > 1 else ""))
    page = head[1].number_input("쪽", 1, pages, 1, key=f"page_{name}", label_visibility="collapsed") if pages > 1 else 1
    if not items:
        st.markdown('<div class="ad-sub">해당하는 질문이 없습니다.</div>', unsafe_allow_html=True)
    for x in items[(page - 1) * PAGE: page * PAGE]:
        render_card(x)


# ---------- 목록 ----------
if hide_done:
    render_section(todo, "todo")
    if not todo:
        st.markdown('<div class="ad-sub">위의 검토 이유를 더 고르거나 "확인한 것 숨기기"를 꺼 보세요.</div>',
                    unsafe_allow_html=True)
else:
    render_section(todo, "todo", "확인 안 함")
    st.divider()
    render_section(done, "done", "확인함")
