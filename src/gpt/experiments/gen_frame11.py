"""실험 11: 선생님 LLM(EXAONE 3.5 7.8B, Ollama)으로 질문 학습 데이터 만들기 — 윈도우 PC 전용

방식 ('틀 먼저'): 해석 틀(정답)은 이 코드가 골고루 뽑고, LLM에게는 "이 틀에 맞는 질문을 써라"만 시킨다.
  -> 정답 라벨은 처음부터 맞고, LLM은 말투가 다양한 질문 쓰기만 한다.
  -> 질문이 틀과 어긋나면(이름 빠짐, 다른 챔피언 섞임, 안 시킨 라인·상향/하향·스킬 언급, 한자 등) 여기서 1차로 버리고,
     frame11_validate.py(맥)가 2차 검증·학습/검증 분할을 한다.
출력: data/frame11/raw/exaone_<tag>.jsonl (gitignore). 한 줄 = {"이전": {"질문","틀"}|null, "질문", "틀", "말투"}
라이선스: EXAONE은 비상업 전용 (ai.md 7장). 서비스에서는 이 데이터로 학습한 MiniGPT만 쓴다.

실행 (프로젝트 루트, Ollama 앱 켜 둔 상태):
  py -3.10 src/gpt/experiments/gen_frame11.py --frames 40 --tag pilot
  py -3.10 src/gpt/experiments/gen_frame11.py --frames 2000 --tag run1 --seed 1
2차(run3, 11/28): 평가 질문이던 말투 예시 교체, 초단문 말투 약 70%, 나+상대 짧은 꼴(순서 반반), 짧은 과거형 패치 vs 미래 예측(데이터밖),
  짧은 이어지는 질문(--follow 0.3, 상대 추가·나 정정·라인 바꿈), 짧은 사이트밖 잡담, 개요 '이름만' 질문.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rag"))
import search as S  # noqa: E402

MODEL = "exaone3.5:7.8b"
URL = "http://127.0.0.1:11434/api/generate"
EMPTY = {"범위": "롤", "의도": [], "챔피언": [], "아이템": None, "스킬": None, "라인": None, "역할군": None,
         "난이도": None, "시점": None, "방향": None, "개수": None, "이어받음": False}

CHAMPS = sorted(S.NAMES)
ALIAS_OF = {}
for a, n in S.ALIASES.items():
    ALIAS_OF.setdefault(n, []).append(a)
ITEMS = sorted(S.ITEM_NAMES)
ITEM_ALIAS_OF = {}
for a, n in S.ITEM_ALIASES.items():
    ITEM_ALIAS_OF.setdefault(n, []).append(a)
PATCHES = ["26.22", "26.21", "26.20", "26.19", "26.18", "26.1", "25.24", "25.S1.1", "14.20", "13.10"]
LANES = ["탑", "정글", "미드", "원딜", "서폿"]
CLASSES = ["탱커", "전사", "마법사", "암살자", "원딜", "서포터"]
SKILLS = ["패시브", "Q", "W", "E", "R", "전체"]

# 2차(run3)부터: 실제 질문은 평균 13자·66%가 12자 이하 → 초단문 말투를 약 70% 뽑는다 (말투, 글자 수 상한)
SHORT_STYLES = [
    ("초단문: 2~4단어, 조사·물음표 생략 (예: '이즈 e 쿨 몇초', '나서스 템트리 ㄱㄱ')", 15),
    ("이름 + 한두 단어 (예: '카이사 너프먹음?', '아리 뭐임')", 12),
    ("명사만 나열, 띄어쓰기 거의 없이 (예: '징크스템트리', '피오라잭스템')", 15),
    ("아주 짧은 반말, 띄어쓰기·물음표 생략", 15),
    ("친구한테 카톡하듯 툭 던지는 짧은 말, 자음 줄임말(ㄱㄱ, ㅇㅇ, ㄹㅇ) 섞어도 됨", 15),
]
SHORT_P = 0.8
LONG_MAX = 40          # 긴 말투도 이 길이까지만 (실제 질문 평균 13자)
STYLES = [
    "띄어쓰기를 거의 안 하는 반말",
    "오타나 자음 줄임말(ㅇㅇ, ㄹㅇ, ㅈㄴ, ㅋㅋ)이 섞인 반말",
    "공손한 존댓말 (~요, ~나요?)",
    "롤을 막 시작한 초보가 쓰는 서툰 말투",
    "상황을 길게 설명하다가 마지막에 묻는 말투",
    "친구한테 카톡하듯 툭 던지는 말투",
    "약간 짜증 섞인 말투",
    "게임 용어(라인전, 딜교, 한타, 왕귀, 코어템)를 섞은 고인물 말투",
    "명령조 (~알려줘, ~해줘, ~정리해)",
]
OFFSITE = ["날씨", "점심 메뉴", "숙제·과제", "다른 게임(발로란트, 오버워치, 메이플)", "연애 고민", "인사·안부",
           "의미 없는 장난이나 가벼운 욕", "주식·코인", "축구·야구 경기", "너는 누구냐(AI 정체)", "코딩 질문",
           "아무 글자나 막 친 것", "롤 프로게이머 사생활·연봉", "컴퓨터 사양·렉 문제"]
OFFSITE_SHORT = "아주 짧은 잡담·감탄·혼잣말 (예: 'ㅋㅋ', '뭐임', '심심해', 'ㅎㅇ', '에휴', '아 졸려')"
FUTURE_TOPIC = "앞으로의 패치 예측 (반드시 '언제', '다음 패치', '예정' 같은 미래 표현을 넣을 것. 예: '언제 상향돼?', '다음 패치에 버프 와?')"
DATAOUT = ["두 스킬을 동시에 쓰는 콤보·상호작용 가능 여부(예: 플래시와 스킬)", "룬 세팅", FUTURE_TOPIC,
           "스킨 추천·가격", "챔피언 대사·성우", "프로 대회 픽·밴", "특정 구간의 피해량 계산", "정글 동선·와드 위치",
           "랭크 티어 올리는 일반 팁", "챔피언 출시일·역사"]

# 나+상대 짧은 꼴: (틀, 나 먼저?) — 나 먼저 / 상대 먼저 반반
# (run5부터 "{M} vs {E}", "{E} 맞라인 {M}"은 누가 나인지 표시가 없는 '맞대결' 꼴이라 여기서 빼고 DUEL_FORMS로 옮김)
VS_FORMS = [("나 {M} 상대 {E}", True), ("내가 {M}고 적이 {E}", True), ("{M}로 {E} 맞라인", True),
            ("상대 {E}, 나 {M}", False), ("{E} 상대로 {M}", False), ("적이 {E}인데 나 {M}", False)]
def format_vs(form, M, E):
    """VS_FORMS 채우기 + 받침에 맞는 조사 (자르반 4세로 / 자르반으로, 키아나고 / 쉔이고)"""
    jong = lambda w: "가" <= w[-1] <= "힣" and (ord(w[-1]) - 0xAC00) % 28 != 0
    ro = lambda w: w + ("으로" if jong(w) and (ord(w[-1]) - 0xAC00) % 28 != 8 else "로")
    return (form.replace("{M}로", ro(M)).replace("{M}고", M + ("이고" if jong(M) else "고"))
            .format(M=M, E=E))


# 맞대결 꼴 (두 챔피언 모두 역할=맞대결, 나·상대 표시 없음). 의도: None = 안 적음 -> [상대법, 템트리]
DUEL_FORMS = {
    None: ["{A} vs {B}", "{A}vs{B}", "{A} 대 {B}", "{A} {B} 맞라인", "{A} {L} vs {B}", "{A} vs {B} {L}"],
    "템트리": ["{A} vs {B} 템트리", "{A} {B} 템트리", "{A}vs{B} 템", "{A} 대 {B} 빌드", "{A} {B} 맞라인 템", "{A} {L} vs {B} 템"],
    "상대법": ["{A}랑 {B} 누가 셈", "{A} {B} 상성", "{A} {B} 라인전", "{A} vs {B} 누가 이김", "{A} 대 {B} 상성", "{A} {B} {L} 라인전"],
}
INTENT_W = {"개요": 7, "스킬": 11, "패치": 16, "이유": 7, "능력치": 6, "템트리": 20, "상대법": 16, "성능": 10,
            "아이템": 7, "추천": 10, "안내": 4}
PAIRS = [("성능", "패치"), ("상대법", "템트리"), ("패치", "이유"), ("스킬", "패치"), ("개요", "템트리"),
         ("템트리", "아이템"), ("개요", "스킬"), ("성능", "템트리"), ("상대법", "스킬"), ("추천", "템트리"),
         ("능력치", "패치"), ("패치", "성능")]


def surface_champ(rng, n):
    return rng.choice(ALIAS_OF[n]) if n in ALIAS_OF and rng.random() < 0.4 else n


def surface_item(rng, n):
    return rng.choice(ITEM_ALIAS_OF[n]) if n in ITEM_ALIAS_OF and rng.random() < 0.5 else n


# ---------------- 1. 틀 뽑기 ----------------
def sample_frame(rng, champs, only_vs=False):
    """-> (틀, 말해야 할 것들 meta). meta: 이름 표기, 프롬프트 설명용 정보
    only_vs: 나+상대가 둘 다 있는 템트리·상대법 틀만 (보충 생성용)"""
    f = dict(EMPTY, 의도=[], 챔피언=[])
    meta = {"names": {}, "topic": None}
    r = 1.0 if only_vs else rng.random()
    if r < 0.15:
        f["범위"] = "사이트밖"
        if rng.random() < 0.4:
            meta["topic"] = OFFSITE_SHORT
            meta["force_style"] = ("아주 짧게 (1~3단어)", 8)
        else:
            meta["topic"] = rng.choice(OFFSITE)
        return f, meta
    if r < 0.25:
        f["범위"] = "데이터밖"
        meta["topic"] = FUTURE_TOPIC if rng.random() < 0.25 else rng.choice(DATAOUT)
        if rng.random() < 0.7:
            c = rng.choice(champs)
            f["챔피언"] = [{"이름": c, "역할": "대상"}]
            meta["names"][c] = surface_champ(rng, c)
            if rng.random() < 0.3:
                f["스킬"] = rng.choice(["Q", "W", "E", "R"])
        return f, meta
    if only_vs:
        intents = rng.choice([["템트리"]] * 5 + [["상대법"]] * 4 + [["상대법", "템트리"], ["템트리", "상대법"]])
    elif rng.random() < 0.2:
        intents = list(rng.choice(PAIRS))
    else:
        intents = [rng.choices(list(INTENT_W), weights=list(INTENT_W.values()))[0]]
    f["의도"] = intents
    I = set(intents)

    def champ(role):
        c = rng.choice([x for x in champs if x not in meta["names"]])
        f["챔피언"].append({"이름": c, "역할": role})
        meta["names"][c] = surface_champ(rng, c)

    if "추천" in I:
        pass                                           # 챔피언 추천은 챔피언 이름 없이
    elif I & {"개요", "성능"} and "템트리" in I:
        champ("나")                                    # "A 어떤 챔이고 템트리는?" — 상대 없이
    elif I & {"템트리", "상대법"}:
        if only_vs or "상대법" in I or rng.random() < 0.75:
            if only_vs or rng.random() < 0.9:
                champ("나")
            champ("상대")
        else:
            champ("나")
        roles = [c["역할"] for c in f["챔피언"]]
        if roles == ["나", "상대"] and rng.random() < (0.85 if only_vs else 0.7):    # 짧은 나·상대 꼴을 정해 줌 (순서 반반)
            form, me_first = rng.choice(VS_FORMS)
            meta["vs_form"] = (form, me_first)
        if rng.random() < 0.35:
            f["라인"] = rng.choice(LANES)
    elif I & {"개요", "스킬", "패치", "이유", "능력치", "성능"}:
        champ("대상")
        if "성능" in I and rng.random() < 0.3:
            f["라인"] = rng.choice(LANES)
    if "스킬" in I:
        f["스킬"] = rng.choice(SKILLS)
    elif I & {"패치", "이유"} and rng.random() < 0.3:
        f["스킬"] = rng.choice(["Q", "W", "E", "R", "패시브"])
    if I & {"패치", "이유"}:
        f["시점"] = rng.choices(["최신", "이력", "패치", None], weights=[5, 2, 2, 1])[0]
        if f["시점"] == "패치":
            f["시점"] = rng.choice(PATCHES)
        if "패치" in I:
            f["방향"] = rng.choices(["상향", "하향", "변경", None], weights=[4, 4, 1, 3])[0]
            if f["시점"] == "이력" and rng.random() < 0.5:
                f["개수"] = "전체"
            if f["방향"] in ("상향", "하향") and I == {"패치"} and f["시점"] in ("최신", None) and rng.random() < 0.8:
                # 짧은 과거형 ("X 상향됐나") — 미래 예측(데이터밖)과 대비
                meta["force_style"] = ("짧은 과거형 질문, 이미 일어난 일을 묻기 (예: 'X 상향됐나', 'X Q 너프먹음?', 'X 버프 받았냐'). "
                                       "'언제', '다음 패치', '예정' 같은 미래 표현은 쓰지 말 것", 15)
    if "성능" in I and f["시점"] is None:
        f["시점"] = "최신"
    if "아이템" in I:
        it = rng.choice(ITEMS)
        f["아이템"] = it
        meta["names"][it] = surface_item(rng, it)
    if "추천" in I:
        if rng.random() < 0.4:
            f["역할군"] = rng.choice(CLASSES)
        if rng.random() < 0.35:
            f["라인"] = rng.choice(LANES)
        if rng.random() < 0.35:
            f["난이도"] = "쉬움"
        if rng.random() < 0.25:
            f["개수"] = "하나"
    if "템트리" in I and rng.random() < 0.15:
        f["개수"] = "전체"
    return f, meta


def follow_frame(rng, prev, champs):
    """이전 틀을 이어받는 다음 질문 틀 (질문에는 이어받은 챔피언 이름을 쓰지 않음)"""
    f = dict(EMPTY, 의도=[], 챔피언=[dict(c) for c in prev["챔피언"]])
    prev_names = [c["이름"] for c in prev["챔피언"]]
    meta = {"names": {}, "topic": None, "inherit": list(prev_names)}
    f["이어받음"] = True
    has_me = any(c["역할"] == "나" for c in f["챔피언"])
    has_enemy = any(c["역할"] == "상대" for c in f["챔피언"])
    PI = set(prev["의도"])
    kind = rng.random()
    if has_me and not has_enemy and kind < 0.3:
        f["의도"] = [i for i in prev["의도"] if i in ("템트리", "상대법")] or ["템트리"]
        c = rng.choice([x for x in champs if x not in prev_names])
        f["챔피언"] = [x for x in f["챔피언"] if x["역할"] == "나"] + [{"이름": c, "역할": "상대"}]
        meta["names"][c] = surface_champ(rng, c)
        f["라인"] = prev["라인"]
        meta["mode"] = "상대 추가 (예: '상대는 X야', 'X 상대면?')"
    elif has_me and has_enemy and kind < 0.3:
        f["의도"] = list(prev["의도"])
        c = rng.choice([x for x in champs if x not in prev_names])
        f["챔피언"] = [{"이름": c, "역할": "나"}] + [x for x in f["챔피언"] if x["역할"] == "상대"]
        meta["names"][c] = surface_champ(rng, c)
        meta["inherit"] = [x["이름"] for x in f["챔피언"] if x["역할"] == "상대"]
        f["라인"], f["개수"] = prev["라인"], prev["개수"]
        meta["mode"] = "앞에서 내 챔피언을 잘못 알아들어서 정정함 (예: '나는 X라니까', '아니 나 X')"
    elif PI and PI <= {"템트리", "상대법", "성능"} and kind < 0.45:
        f["의도"] = list(prev["의도"])
        f["라인"] = rng.choice([x for x in LANES if x != prev["라인"]])
        f["시점"], f["개수"] = prev["시점"], prev["개수"]
        meta["mode"] = f"라인만 바꿔 다시 물음 (예: '{f['라인']}이면?', '{f['라인']}은?')"
    elif kind < 0.65:
        f["의도"] = ["스킬"]
        f["스킬"] = rng.choice(SKILLS)
        f["챔피언"] = [{"이름": c["이름"], "역할": "대상"} for c in f["챔피언"][:1]]
        meta["mode"] = "그 챔피언의 스킬로 화제를 바꿈 (예: 'Q는?', '궁은', '스킬 다')"
    elif kind < 0.82:
        f["의도"] = [rng.choice(["패치", "성능", "능력치", "개요"])]
        f["챔피언"] = [{"이름": c["이름"], "역할": "대상"} for c in f["챔피언"][:1]]
        if f["의도"] == ["패치"]:
            f["시점"] = "최신"
        if f["의도"] == ["성능"]:
            f["시점"] = "최신"
        meta["mode"] = "같은 챔피언에 대해 다른 걸 물음 (예: '패치는?', '요즘 좋음?', '체력은', '뭐하는 챔임')"
    else:
        f["의도"] = ["템트리"]
        f["챔피언"] = [{"이름": c["이름"], "역할": "나" if c["역할"] in ("나", "대상") else c["역할"]}
                     for c in f["챔피언"]]
        if rng.random() < 0.5:
            f["라인"] = rng.choice(LANES)
        else:
            f["개수"] = "전체"
        f["라인"] = f["라인"] or prev["라인"]
        meta["mode"] = "템트리로 화제를 바꿈 (예: '그럼 템트리는', '템 뭐감')"
    meta["inherit"] = [n for n in meta["inherit"] if any(c["이름"] == n for c in f["챔피언"])]
    meta["prev_names"] = prev_names
    meta["same_intent"] = bool(f["의도"]) and set(f["의도"]) <= PI
    meta["force_style"] = ("아주 짧은 후속 질문, 12자 이내", 15)
    return f, meta


# ---------------- 2. 틀 -> 프롬프트 ----------------
def describe(f, meta):
    nm = meta["names"]
    if meta.get("lines"):                                   # --only-hard: 유형별로 직접 쓴 설명
        return list(meta["lines"]), list(meta.get("must", []))
    if f["범위"] == "사이트밖":
        return [f"롤 정보 사이트에 잘못 들어온, 롤과 전혀 상관없는 말: 주제 '{meta['topic']}'",
                "롤·게임·챔피언·패치 얘기는 한 마디도 섞지 말 것"], []
    lines, must = [], []
    who = {c["역할"]: c["이름"] for c in f["챔피언"]}
    inherit = set(meta.get("inherit", []))
    for c in f["챔피언"]:
        if c["이름"] in inherit:
            continue
        must.append(nm[c["이름"]])
    if f["범위"] == "데이터밖":
        t = f"롤 질문이지만 주제는 '{meta['topic']}'"
        if who.get("대상"):
            t += f", 챔피언 '{nm[who['대상']]}'"
            if f["스킬"]:
                t += f"의 {f['스킬']} 스킬"
        return [t], must

    def n(role):
        x = who.get(role)
        return None if x is None else ("(앞 대화의 그 챔피언, 이름은 쓰지 말 것)" if x in inherit else f"'{nm[x]}'")
    if n("나"):
        lines.append(f"질문자는 {n('나')}를 플레이한다")
    if n("상대"):
        lines.append(f"상대(적, 맞라인)는 {n('상대')}")
    if n("대상"):
        lines.append(f"물어보는 챔피언은 {n('대상')}")
    if meta.get("vs_form"):
        form = format_vs(meta["vs_form"][0], nm[who["나"]], nm[who["상대"]])
        lines.append(f"나와 상대는 '{form}' 꼴로 짧게 쓴다 (두 이름의 순서를 바꾸지 말 것)")
    D = {"개요": "그 챔피언이 어떤 챔피언인지 (소개·컨셉)",
         "스킬": "스킬 설명",
         "패치": "패치에서 무엇이 바뀌었는지",
         "이유": "왜 그렇게 바뀌었는지 (패치 이유)",
         "능력치": "기본 능력치 (체력·공격력·방어력·사거리 등)",
         "템트리": "아이템 빌드·템트리(구매 순서)",
         "상대법": "상대를 어떻게 이기는지, 상대하는 요령",
         "성능": "요즘 좋은지·OP인지·승률",
         "아이템": f"아이템 '{nm.get(f['아이템'], '')}'의 효과나 조합",
         "추천": "어떤 챔피언을 할지 추천",
         "안내": "이 사이트가 뭐 하는 곳인지, 무엇을 물어볼 수 있는지"}
    lines.append("묻는 것: " + " + ".join(D[i] for i in f["의도"]))
    if f["아이템"]:
        must.append(nm[f["아이템"]])
    if f["스킬"]:
        lines.append("스킬: " + {"전체": "스킬 전부", "패시브": "패시브"}.get(f["스킬"], f"{f['스킬']} 스킬 (Q/W/E/R 글자 또는 R이면 '궁'이라고 써도 됨)"))
    if f["라인"]:
        lines.append(f"라인: {f['라인']} (질문에 라인을 꼭 언급)")
    if f["역할군"]:
        lines.append(f"원하는 역할군: {f['역할군']}")
    if f["난이도"]:
        lines.append("질문자는 롤 초보라서 쉬운 챔피언을 원함")
    if f["시점"]:
        lines.append("시점: " + {"최신": "최근·요즘·이번 패치", "이력": "예전부터 지금까지의 변화 전부"}.get(f["시점"], f"{f['시점']} 패치 (이 번호를 그대로 쓸 것)"))
        if f["시점"] not in ("최신", "이력"):
            must.append(f["시점"])
    if f["방향"]:
        lines.append("방향: " + {"상향": "상향(버프)됐는지", "하향": "하향(너프)됐는지", "변경": "뭐가 바뀌었는지 (좋아졌는지 나빠졌는지는 말하지 말 것)"}[f["방향"]])
    if f["개수"]:
        lines.append("개수: " + {"하나": "딱 하나만 원함", "전체": "전부·다 알려 달라고 함"}[f["개수"]])
    if f.get("이어받음"):
        lines.append(f"이건 앞 질문에 이어서 묻는 짧은 후속 질문이다 ({meta.get('mode')}). 앞에서 말한 챔피언 이름은 다시 쓰지 않는다")
    return lines, must


def pick_style(rng, meta):
    """-> (말투 설명, 글자 수 상한 또는 None)"""
    if meta.get("force_style"):
        return meta["force_style"]
    if rng.random() < SHORT_P:
        return rng.choice(SHORT_STYLES)
    return rng.choice(STYLES), LONG_MAX


def build_prompt(f, meta, style, k, prev_q=None):
    lines, must = describe(f, meta)
    style, maxlen = style
    # 말투 예시는 실제 평가 질문(logs/eval/exp11_real_gold.jsonl)에 없는 문장만 쓴다 (평가 오염 방지)
    p = ["너는 한국 리그 오브 레전드(롤) 유저다. 롤 정보 사이트 채팅창에 입력할 질문을 쓴다.",
         "유저 질문은 대부분 아주 짧다. 말투 예: '이즈 e 쿨 몇초야', '나서스 템트리 ㄱㄱ', '피오라 vs 잭스 템 어케감', '징수의 총 가격 얼마', '카이사 너프먹음?'",
         "(예시의 챔피언 이름은 쓰지 말고, 아래 내용의 이름만 쓸 것)",
         ""]
    if prev_q:
        p.append(f"앞 질문: \"{prev_q}\"")
    p.append("이번 질문의 내용:")
    p += [f"- {x}" for x in lines]
    if must:
        p.append("- 반드시 이 표기를 그대로 포함: " + ", ".join(f"'{m}'" for m in must))
    p += ["",
          "규칙: 위 내용에 없는 다른 챔피언·아이템 이름, 라인, 상향/하향 여부는 쓰지 않는다. 한국어만 쓴다.",
          f"말투: {style}" + (f" — 질문 하나당 {maxlen}자 이내" if maxlen else ""),
          f"서로 표현이 다른 질문 {k}개를 한 줄에 하나씩, 번호·따옴표·설명 없이 질문만 출력하라."]
    return "\n".join(p), must


def call(prompt, seed, temp=0.8):     # 2차부터 0.95 -> 0.8 (지어낸 단어 줄이기)
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "top_p": 0.95, "seed": seed, "num_predict": 300}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read().decode())["response"]


# ---------------- 3. 1차 걸러내기 ----------------
BAD_CHARS = re.compile(r"[一-鿿぀-ヿ]")
LANE_WORDS = re.compile(r"탑|정글|미드|원딜|서폿|서포터|바텀|봇|원거리")
DIR_WORDS = re.compile(r"상향|하향|버프|너프|좋아졌|나빠졌|약해졌|강해졌|쎄졌|세졌")
SKILL_WORDS = re.compile(r"(?<![A-Za-z])[QWERqwer](?![A-Za-z])|궁|패시브|스킬")
FUTURE = re.compile(r"언제|다음 ?패치|예정|될까|되려나|해줄까|해주나|해줬으면|오려나|앞으로")
ENEMY_CUE = re.compile(r"상대|적|vs|VS|맞라인|라인전|카운터|만나|잡")
CLEAN = re.compile(r"^\s*(?:[-*•]|\d+[.)](?!\d)|\d+(?=\s))\s*")   # "3.질문"처럼 번호 뒤 띄어쓰기 없는 것도


NAME_RE = re.compile("|".join(re.escape(n) for n in sorted(S.NAMES, key=len, reverse=True)))


def find_names(q):
    """서비스 사전과 같은 방식(normalize 후 긴 이름 먼저)으로 찾은 챔피언 이름"""
    return set(NAME_RE.findall(S.normalize(q)))


ITEM_RE = re.compile("|".join(re.escape(n) for n in sorted(set(S.ITEM_NAMES) | set(S.ITEM_ALIASES), key=len, reverse=True)))
LOL_WORDS = re.compile(r"롤|리그 ?오브|챔피언|챔프|패치|스킬|아이템|템트리|라인|정글|미드|원딜|서폿|탑|랭크|티어|협곡|칼바람|한타|바론|용|소환사")
# 틀에 있는 내용이 질문에 실제로 드러나는지 (없으면 라벨과 질문이 어긋나므로 버림)
INTENT_CUE = {
    "개요": r"누구|뭐야|뭐임|뭔데|머야|머임|뭐냐|머냐|무슨 ?챔|어떤 ?챔|소개|컨셉|콘셉|특징|어떤 ?역할|정체|플레이 ?스타일|뭐 ?하는|뭐함|알려|설명|정보|어떤 ?애|무슨 ?애",
    "스킬": r"스킬|(?<![A-Za-z])[QWERqwer](?![A-Za-z])|궁|패시브|기술",
    "패치": r"패치|바뀌|바꼈|바뀠|바뀐|변경|변했|변화|달라|상향|하향|버프|너프|업데이트|조정|수정",
    "이유": r"왜|이유|의도|어째서",
    "능력치": r"능력치|체력|공격력|방어력|마법 ?저항|사거리|이속|이동 ?속도|스탯|마나|공속|공격 ?속도",
    "템트리": r"템|아이템|빌드|코어|뭐 ?사|뭘 ?사|구매|뭐감|뭐 ?가|뭐 ?올|뭐 ?먼저 ?사|먼저 ?사",
    "상대법": r"상대|이기|이겨|카운터|vs|VS|잡|대처|라인전|맞라인|대응|파훼|압도|만나|적|어케 ?해|어떻게 ?해|팁|운영|요령|잘하|노하우|상성|이김|이길|승자|유리|우위|앞서|누가 ?(셈|세|쎔|강)",
    "성능": r"좋|OP|op|승률|쎄|세\?|센가|강하|강해|사기|메타|성능|티어|할만|할 ?만|어때|어떰|구리|구림|쓰레기|쓸만|1티어|떡상|떡락|괜찮|ㄱㅊ|망했|망함|해볼만|버려",
    "아이템": r".",
    "추천": r"추천|할까|뭐 ?하|뭐 ?할|골라|픽|고를|할만한|할 ?만한",
    "안내": r"사이트|여기|뭐 ?하는|뭘 ?물어|무엇을|기능|사용법|어떻게 써",
}
SKILL_CUE = {"Q": r"(?<![A-Za-z])[Qq](?![A-Za-z])", "W": r"(?<![A-Za-z])[Ww](?![A-Za-z])", "E": r"(?<![A-Za-z])[Ee](?![A-Za-z])",
             "R": r"(?<![A-Za-z])[Rr](?![A-Za-z])|궁", "패시브": r"패시브", "전체": r"스킬"}
FIELD_CUE = {
    "난이도": {"쉬움": r"초보|처음|쉬운|쉽|입문|뉴비|롤린이|시작"},
    "개수": {"하나": r"하나|한 ?명|한 ?개|1개|1명|딱", "전체": r"전부|다 |다\?|다$|전체|모든|싹|모두|총정리|풀"},
    "시점": {"이력": r"예전|옛날|과거|처음|역사|지금까지|그동안|변천|이력|여태|출시|역대|쭉"},
    "방향": {"상향": r"상향|버프|좋아졌|강해졌|쎄졌|세졌|좋게", "하향": r"하향|너프|나빠졌|약해졌|구려졌|안 ?좋아"},
}
CLASS_CUE = {"탱커": r"탱", "전사": r"전사|브루저", "마법사": r"마법사|메이지|AP", "암살자": r"암살자|어쌔|암살",
             "원딜": r"원딜|원거리", "서포터": r"서포터|서폿|서포트"}
LANE_CUE = {"탑": r"탑", "정글": r"정글", "미드": r"미드", "원딜": r"원딜|바텀|봇", "서폿": r"서폿|서포터|서포트"}


def check(q, f, meta, must, maxlen=None):
    if not q or len(q) < 2 or len(q) > 120:
        return "길이"
    if maxlen and len(q) > maxlen + 3 + 5 * max(0, len(must) - 1):   # 짧게 시킨 말투인데 길게 씀
        return "너무 김"
    if BAD_CHARS.search(q):
        return "한자·일본어"
    if f["범위"] == "사이트밖":
        if meta.get("allow_lol"):                           # "너 혹시 X임?" — 챗봇에게 묻는 말이라 이름이 있어도 됨
            return None if all(m in q for m in must) and not re.search(r"템|스킬|패치|라인|빌드|승률", q) else "표기 빠짐·롤 정보"
        return "롤 얘기 섞임" if (find_names(q) or LOL_WORDS.search(q)) else None
    for m in must:
        if m not in q:
            return "표기 빠짐"
    items = set(ITEM_RE.findall(q))
    want_items = {meta["names"][f["아이템"]], f["아이템"]} if f["아이템"] else set()
    if items - want_items and not any(i in w for i in items for w in want_items):
        return "다른 아이템"
    if f["범위"] == "데이터밖" and meta.get("topic") == FUTURE_TOPIC and not FUTURE.search(q):
        return "미래 표현 없음"
    if f["범위"] == "롤":
        if FUTURE.search(q):
            return "미래 표현 (데이터밖과 헷갈림)"
        for i in ([] if meta.get("same_intent") else f["의도"]):   # 같은 의도 이어 묻기("탑이면?")는 단서어 없어도 됨
            if not re.search(INTENT_CUE[i], q):
                return f"의도 '{i}' 안 드러남"
        if meta.get("vs_form"):
            me = meta["names"][next(c["이름"] for c in f["챔피언"] if c["역할"] == "나")]
            en = meta["names"][next(c["이름"] for c in f["챔피언"] if c["역할"] == "상대")]
            if me in q and en in q and (q.find(me) < q.find(en)) != meta["vs_form"][1]:
                return "나·상대 순서 바뀜"
        if meta.get("need") and not re.search(meta["need"], re.sub("|".join(map(re.escape, must)) or "^$", " ", q)):   # 이름 속 글자(쉬바"나") 제외
            return "정정 꼴 아님"
        if meta.get("last_me"):                             # 맞라인 서술: 끝에서 템을 묻는 챔피언(나)이 상대보다 뒤에
            a, e = meta["last_me"]
            if q.rfind(a) < q.rfind(e):
                return "끝에 묻는 챔피언이 상대"
        roles = {c["역할"] for c in f["챔피언"]}
        if {"나", "상대"} <= roles and not f.get("이어받음") and not ENEMY_CUE.search(q):
            return "누가 상대인지 안 드러남"
        if f["역할군"] and not re.search(CLASS_CUE[f["역할군"]], q):
            return "역할군 안 드러남"
        for k, cues in FIELD_CUE.items():   # 같은 의도 이어 묻기는 앞 틀에서 물려받은 칸이라 단서어 없어도 됨
            if f[k] in cues and not meta.get("same_intent") and not re.search(cues[f[k]], q):
                return f"{k} 안 드러남"
    if f["스킬"] and not re.search(SKILL_CUE[f["스킬"]], q):
        return "스킬 안 드러남"
    if f["라인"] and not re.search(LANE_CUE[f["라인"]], q):
        return "라인 안 드러남"
    want = {c["이름"] for c in f["챔피언"] if c["이름"] not in set(meta.get("inherit", []))}
    got = find_names(q)
    # 이름 안에 다른 이름이 들어 있는 경우(예: '리 신' 안의 '신') 대비: 원하는 이름의 일부인 것은 무시
    extra = {g for g in got - want if not any(g in w for w in want)}
    if extra:
        return "다른 챔피언"
    if want - got:
        return "사전이 이름 못 찾음"
    if f["범위"] == "데이터밖":
        return None
    if f["라인"] is None and LANE_WORDS.search(q) and f["역할군"] not in ("원딜", "서포터"):
        return "안 시킨 라인"
    if f["방향"] is None and DIR_WORDS.search(q) and "성능" not in f["의도"]:
        return "안 시킨 방향"
    if f["스킬"] is None and SKILL_WORDS.search(q) and not set(f["의도"]) & {"개요"}:
        return "안 시킨 스킬"
    return None


# ---------------- 어려운 유형만 (--only-hard, run4) ----------------
# 실험 11d의 남은 오류 유형을 겨냥. 실제 사용자 질문(로그·평가 세트)은 쓰지 않고, 사용자가 예로 든 챔피언도 피한다.
HARD_AVOID = {"가렌", "티모", "케일", "마스터 이", "잭스", "카직스", "르블랑", "케넨", "애니", "이즈리얼", "리븐", "케이틀린",
              "피즈", "카밀", "피오라", "에코", "자야"}
HARD_TYPES = ["정정 이어받기", "스킬 패치", "범위 경계", "맞라인 서술", "스킬 목표 템트리", "팁·이력"]
MECHANICS = ["평타 캔슬(평캔)이 되는지", "애니메이션 캔슬 방법", "스킬 사거리·판정 세부(어디까지 맞는지)",
             "스킬끼리 동시에 쓰는 조작·버그", "점멸과 스킬 동시 사용 판정"]
BOT_TOPICS = ["너는 누구야, 사람이야 AI야 (챗봇 자신에게 묻기)", "너 이름 뭐야, 누가 만들었어 (챗봇 자신에게 묻기)"]
SYMBOLS = ["?", "??", "..", "...", "ㅁ", "ㅇ", "ㄱ", "!", ".", "ㅁㄴㅇㄹ", ";;", "ㅡㅡ", "?!", "ㅎ", "/"]
ONE_WORD = ["챔피언", "추천", "챔추천", "챔피언 추천", "추천좀", "챔프", "챔 추천", "추천해"]
STAT_GOALS = ["공속", "방관", "치명타", "쿨감", "체력", "AP"]


def _jong(w):
    """받침 있으면 True (조사 고르기: 베인이야 / 퀸이야 / 럭스야)"""
    c = w[-1]
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def _ya(w):
    return w + ("이야" if _jong(w) else "야")


def _ranikka(w):
    return w + ("이라니까" if _jong(w) else "라니까")


def _champ(rng, champs, used):
    c = rng.choice([x for x in champs if x not in used and x not in HARD_AVOID])
    used.add(c)
    return c


def _prev_row(kind, A=None, E=None, rng=None):
    """이어받기용 앞 질문 (코드가 쓰는 짧은 틀 문장 + 그 틀). MiniGPT 입력은 앞 '틀'만 쓰므로 문장은 맥락용."""
    f = dict(EMPTY, 의도=[], 챔피언=[])
    if kind == "템트리_나":
        f["의도"], f["챔피언"] = ["템트리"], [{"이름": A, "역할": "나"}]
        q = rng.choice([f"{A} 템트리", f"{A} 템 뭐감", f"{A} 빌드 알려줘"])
    elif kind == "상대법_나":
        f["의도"], f["챔피언"] = ["상대법"], [{"이름": A, "역할": "나"}]
        q = rng.choice([f"{A} 라인전 어케함", f"{A}로 이기는 법"])
    elif kind == "개요":
        f["의도"], f["챔피언"] = ["개요"], [{"이름": A, "역할": "대상"}]
        q = rng.choice([f"{A} 뭐임", f"{A}", f"{A} 어떤 챔이야"])
    elif kind == "템트리_상대":
        f["의도"], f["챔피언"] = ["템트리"], [{"이름": E, "역할": "상대"}]
        q = rng.choice([f"상대 {E} 템트리", f"{E} 상대 템 뭐감"])
    elif kind == "상대법_상대":
        f["의도"], f["챔피언"] = ["상대법"], [{"이름": E, "역할": "상대"}]
        q = rng.choice([f"{E} 어떻게 이김", f"{E} 상대하는 법"])
    else:  # 나상대
        f["의도"] = [rng.choice(["템트리", "상대법"])]
        f["챔피언"] = [{"이름": A, "역할": "나"}, {"이름": E, "역할": "상대"}]
        q = rng.choice([f"나 {A} 상대 {E} " + ("템" if f["의도"] == ["템트리"] else "어케 이김"),
                        f"{E} 상대로 {A} " + ("템트리" if f["의도"] == ["템트리"] else "라인전")])
    return {"질문": q, "틀": f}


def hard_job(rng, champs):
    """-> dict(f, meta, prev, style, k, fixed, type). fixed = 코드가 직접 쓰는 질문 목록 (LLM 호출 없음)"""
    t = rng.choice(HARD_TYPES)
    f = dict(EMPTY, 의도=[], 챔피언=[])
    meta = {"names": {}, "topic": None, "hard": t}
    used, prev, fixed, k, extra = set(), None, None, 4, []
    style = rng.choice(SHORT_STYLES)

    def add(c, role):
        f["챔피언"].append({"이름": c, "역할": role})
        meta["names"][c] = surface_champ(rng, c)

    if t == "정정 이어받기":
        A, E, B = _champ(rng, champs, used), _champ(rng, champs, used), _champ(rng, champs, used)
        f["이어받음"], k = True, 3
        style = ("아주 짧은 후속 질문, 12자 이내", 15)
        if rng.random() < 0.5:                                    # 상대 추가: 앞 챔피언이 '나'가 됨
            pk = rng.choice(["템트리_나", "상대법_나", "개요"])
            prev = _prev_row(pk, A=A, rng=rng)
            f["의도"] = ["템트리"] if pk == "템트리_나" else ["상대법"] if pk == "상대법_나" else [rng.choice(["템트리", "상대법"])]
            f["챔피언"] = [{"이름": A, "역할": "나"}]
            add(E, "상대")
            meta["inherit"] = [A]
            meta["same_intent"] = pk != "개요"
            e = meta["names"][E]
            ask = "" if meta["same_intent"] else (f" 템 뭐감" if f["의도"] == ["템트리"] else f" 어케 이김")
            meta["lines"] = ["앞 질문에 이어서, 내 상대(맞라인) 챔피언만 새로 알려 주는 아주 짧은 말",
                             f"꼴 예: '상대는 {e}야{ask}', '{e} 상대면{ask or '?'}', '상대 {e}인데{ask or ' 어케함'}'",
                             "꼴 예와 거의 같은 모양으로, '상대'라는 말을 꼭 넣을 것. 앞 질문의 챔피언 이름은 다시 쓰지 말 것, 다른 챔피언·라인·스킬도 쓰지 말 것"]
            meta["must"] = [e]
            meta["need"] = r"상대|적|맞라인"
            tail = "" if meta["same_intent"] else ask
            extra = [f"상대는 {_ya(e)}{tail}", f"상대 {e}{tail}", f"{e} 상대면?" if not tail else f"{e} 상대면{tail}",
                     f"상대 {e}인데{tail or ' 어떡함'}", f"적 {e}임{tail}", f"맞라인 {e}{tail}", f"상대가 {_ranikka(e)}{tail}"]
        else:                                                     # 나 정정: 앞 상대는 그대로, 내 챔피언을 고침
            pk = rng.choice(["템트리_상대", "상대법_상대", "나상대"])
            prev = _prev_row(pk, A=A, E=E, rng=rng)
            f["의도"] = list(prev["틀"]["의도"])
            f["챔피언"] = [x for x in prev["틀"]["챔피언"] if x["역할"] == "상대"]
            add(B, "나")
            meta["inherit"] = [E]
            meta["same_intent"] = True
            b = meta["names"][B]
            meta["lines"] = ["앞 질문에 이어서, 내(질문자) 챔피언을 알려 주거나 정정하는 아주 짧은 말",
                             f"꼴 예: '나는 {b}', '나 {b}라니까', '아니 내가 {b}임', '난 {b}야'",
                             "꼴 예와 거의 같은 모양으로, '나'·'난'·'내가' 중 하나를 꼭 넣을 것. 앞 질문의 챔피언 이름은 다시 쓰지 말 것, 다른 챔피언·라인·스킬도 쓰지 말 것"]
            meta["must"] = [b]
            meta["need"] = r"나|난|내가"
            extra = [f"나는 {b}", f"나 {_ranikka(b)}", f"아니 내가 {b}임", f"난 {_ya(b)}", f"나 {b}임", f"아니 나 {b}", f"내가 {b}" + ("이라고" if _jong(b) else "라고")]
        meta["prev_names"] = [c["이름"] for c in prev["틀"]["챔피언"]]
        meta["prev_surface"] = {n: n for n in meta["prev_names"]}
    elif t == "스킬 패치":
        add(_champ(rng, champs, used), "대상")
        f["의도"], f["스킬"] = ["패치"], rng.choice(["Q", "W", "E", "R", "패시브"])
        f["방향"], f["시점"] = rng.choice(["상향", "하향"]), "최신"
        a, s = meta["names"][f["챔피언"][0]["이름"]], {"R": "궁", "패시브": "패시브"}.get(f["스킬"], f["스킬"])
        w = ["상향", "버프"] if f["방향"] == "상향" else ["너프", "하향"]
        meta["lines"] = [f"'{a}'의 {f['스킬']} 스킬이 최근 패치에서 {f['방향']}됐는지 묻는 아주 짧은 말",
                         f"꼴 예: '{a} {s}{w[0]}', '{a} {s} {w[1]}됨?', '{a} {s} {w[0]}먹음'",
                         f"'{w[0]}' 또는 '{w[1]}'를 꼭 넣을 것('변경'·'변화'는 쓰지 말 것). 미래 표현(언제·다음 패치) 금지, 다른 스킬·챔피언 쓰지 말 것"]
        meta["must"] = [a]
        style = ("2~3단어로 아주 짧게", 14)
    elif t == "범위 경계":
        sub = rng.choice(["롤 스킬 정보", "데이터밖 조작", "데이터밖 미래", "사이트밖 챗봇", "사이트밖 기호"])
        meta["sub"] = sub
        if sub == "롤 스킬 정보":
            add(_champ(rng, champs, used), "대상")
            f["의도"], f["스킬"] = ["스킬"], rng.choice(["Q", "W", "E", "R", "패시브"])
            style = ("이름 + 스킬 + 한 단어로 아주 짧게 (예: 'X R 정보', 'X 패시브 뭐임', 'X q')", 10)
        elif sub.startswith("데이터밖"):
            f["범위"] = "데이터밖"
            add(_champ(rng, champs, used), "대상")
            meta["topic"] = FUTURE_TOPIC if sub == "데이터밖 미래" else rng.choice(MECHANICS)
            style = ("짧은 반말 (예: 'X 평캔 되냐', 'X 애니캔 방법', 'X 사거리 판정')", 15)
        elif sub == "사이트밖 챗봇":
            f["범위"] = "사이트밖"
            if rng.random() < 0.5:                       # "너 혹시 X임?" — 챔피언 이름이 있어도 챗봇에게 묻는 말
                c = _champ(rng, champs, used)
                meta["names"][c] = surface_champ(rng, c)
                meta["allow_lol"] = True
                meta["lines"] = [f"롤 정보 사이트의 챗봇 자신에게 '너 혹시 {meta['names'][c]}냐'고 장난으로 묻는 말 (롤 정보는 안 물음)"]
                meta["must"] = [meta["names"][c]]
            else:
                meta["topic"] = rng.choice(BOT_TOPICS)
            style = ("아주 짧은 반말 (예: '너 누구야', '너 혹시 사람임?')", 12)
        else:
            f["범위"] = "사이트밖"
            fixed = rng.sample(SYMBOLS, 2)
    elif t == "맞라인 서술":
        A, E = _champ(rng, champs, used), _champ(rng, champs, used)
        f["의도"] = ["템트리"]
        add(A, "나")
        add(E, "상대")
        a, e = meta["names"][A], meta["names"][E]
        meta["lines"] = [f"상황: 질문자는 '{a}'를 하는데 맞라인(상대)으로 '{e}'가 나왔다",
                         f"맞라인 상황을 먼저 말하고, 문장 끝에서 '{a}' 템트리(뭐 살지)를 묻는다. '{a}' 이름은 두 번 나와도 된다",
                         "꼴 예: 'X 맞라인에 Y 나왔는데 X 템 뭐감', 'Y 맞라인 걸렸는데 X 템트리', 'X인데 맞라인 Y임 템 뭐가 좋음'"]
        meta["must"] = [a, e]
        meta["last_me"] = (a, e)
        style = ("짧은 반말, 띄어쓰기 대충", 30)
    elif t == "스킬 목표 템트리":
        add(_champ(rng, champs, used), "나")
        f["의도"] = ["템트리"]
        a = meta["names"][f["챔피언"][0]["이름"]]
        if rng.random() < 0.75:
            f["스킬"] = rng.choice(["Q", "W", "E", "R"])
            s = {"R": "궁", "Q": "Q", "W": "W", "E": "E"}[f["스킬"]]
            meta["lines"] = [f"질문자는 '{a}'를 한다", f"{f['스킬']} 스킬(쿨·피해·사거리 등)을 좋게 하는 템·템트리를 묻는다. 다른 스킬은 쓰지 말 것",
                             f"꼴 예: '{a} {s} 쿨 줄이는 템', '{a} {s} 딜 올리는 템트리', '{a} {s.lower()} 강화템'"]
        else:
            g = rng.choice(STAT_GOALS)
            meta["lines"] = [f"질문자는 '{a}'를 한다", f"'{g}' 위주 템트리를 묻는다. 스킬 이름(QWER·궁·스킬)은 쓰지 말 것",
                             f"꼴 예: 'X {g} 템트리', 'X {g}템 뭐감'"]
        meta["must"] = [a]
        style = ("아주 짧은 반말 (15자 안팎)", 18)
    else:  # 팁·이력
        sub = rng.choice(["팁", "이력", "단어 하나"])
        meta["sub"] = sub
        if sub == "팁":
            add(_champ(rng, champs, used), "나")
            f["의도"] = ["상대법"]
            a = meta["names"][f["챔피언"][0]["이름"]]
            meta["lines"] = [f"질문자는 '{a}'를 한다", "그 챔피언을 잘하는 팁·운영 요령을 묻는다 (상대 챔피언 이름 없이)",
                             "꼴 예: 'X 팁', 'X 후반 운영 꿀팁', 'X 한타 요령', 'X 잘하는법'"]
            meta["must"] = [a]
            style = ("아주 짧게 (2~4단어)", 12)
        elif sub == "이력":
            add(_champ(rng, champs, used), "대상")
            f["의도"], f["시점"], f["방향"], f["개수"] = ["패치"], "이력", "변경", "전체"
            if rng.random() < 0.6:
                f["스킬"] = rng.choice(["Q", "W", "E", "R", "패시브"])
            a = meta["names"][f["챔피언"][0]["이름"]]
            s = {"R": "궁", "패시브": "패시브"}.get(f["스킬"], f["스킬"]) if f["스킬"] else ""
            meta["lines"] = [f"'{a}'{(' ' + s) if s else ''} 패치 변화를 옛날부터 지금까지 전부 정리해 달라는 말",
                             "'옛날부터', '역대', '지금까지', '여태' 중 하나와 '다', '전부', '싹' 중 하나, '바뀐'·'패치' 중 하나를 꼭 넣을 것" + (f", 스킬 '{s}'도 꼭 넣을 것" if s else ", 스킬 이름은 쓰지 말 것") + ". 좋아졌는지 나빠졌는지는 말하지 말 것",
                             f"꼴 예: '{a}{(' ' + s) if s else ''} 옛날부터 바뀐 거 다 정리', '{a} 역대 패치 전부'"]
            meta["must"] = [a]
            style = ("짧은 반말", 20)
        else:
            f["의도"] = ["추천"]
            fixed = [rng.choice(ONE_WORD)]
            if rng.random() < 0.6:                       # 앞 대화가 있어도 이어받지 않음
                A, E = _champ(rng, champs, used), _champ(rng, champs, used)
                prev = _prev_row(rng.choice(["템트리_나", "개요", "나상대", "상대법_상대"]), A=A, E=E, rng=rng)
    # 정정 이어받기는 EXAONE이 꼴을 잘 안 따라서 코드 정형 문장 2개를 섞음
    return {"f": f, "meta": meta, "prev": prev, "style": style, "k": k, "fixed": fixed, "type": t,
            "extra": rng.sample(extra, 2) if extra else []}


def parse_lines(text):
    out = []
    for ln in text.splitlines():
        ln = CLEAN.sub("", ln).replace("'", "").replace('"', "").strip()
        if ln:
            out.append(ln)
    return out


def run_hard(rng, a, i, fo, why):
    """--only-hard 틀 하나 -> 통과 줄 수. 질문 줄의 '말투' 칸에 'hard:유형'을 적어 둠"""
    job = hard_job(rng, CHAMPS)
    f, meta, prev, t = job["f"], job["meta"], job["prev"], job["type"]
    tag = f"hard:{t}" + (f"/{meta['sub']}" if meta.get("sub") else "")
    if job["fixed"]:                                     # 기호·단어 하나: 코드가 직접 씀
        qs, style, must = job["fixed"], ("", None), []
    else:
        style = job["style"]
        prompt, must = build_prompt(f, meta, style, job["k"], prev_q=prev and prev["질문"])
        try:
            qs = parse_lines(call(prompt, a.seed * 100000 + i))[:job["k"]]
        except Exception as e:
            why["호출 실패"] += 1
            print("호출 실패", e, flush=True)
            return 0
    qs = list(qs) + job["extra"]
    n = 0
    for q in qs:
        bad = None if job["fixed"] else check(q, f, meta, must, style[1])
        if not bad and f.get("이어받음") and any(nm in S.normalize(q) for nm in meta["prev_names"]):
            bad = "이어받을 이름을 다시 씀"
        why[f"{t}: " + (bad or "통과")] += 1
        if bad and a.show_bad:
            print(f"  버림 [{t}|{bad}] {q}" + (f"  <- {prev['질문']}" if prev else ""), flush=True)
        if not bad:
            fo.write(json.dumps({"이전": prev, "질문": q, "틀": f, "말투": tag}, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------- 맞대결 (run5) ----------------
# 나·상대 표시 찾기: frame11_validate.vs_ambiguous와 같은 규칙 (윈도우엔 torch가 없어 그 파일을 못 불러와서 옮겨 씀 — 바꾸면 둘 다)
ME_CUE = re.compile(r"(^|\s)(나|난|나는|내가|제가|저는|저|내)\s|(^|\s)(나|난|저)$")
DUEL_AVOID = {"아이번", "유미"}           # 평가 세트에 있는 조합 — 예시·생성에서 뺌


def has_mark(q, a, b):
    """질문에 누가 나·상대인지 표시가 하나라도 있으면 True (a·b는 정식 이름, 어느 쪽이 나든 상관없이 봄)"""
    t = S.normalize(q)
    if a not in t or b not in t:
        return True                                   # 한쪽 이름이 없으면 맞대결 문장이 아님
    for me, en in ((a, b), (b, a)):
        e, m = re.escape(en), re.escape(me)
        if any([ME_CUE.search(t),
                re.search(r"(상대|적|적군)(이|가|는|은)?\s*(챔프?|챔피언)?\s*" + e, t),
                re.search(e + r"\s*(이|가|를|을)?\s*(상대(?!법)|나왔|나오|만나|만났|잡|걸렸|걸리)", t),
                re.search(e + r"(이?랑|와|과)\s*(맞라인|붙|만나|상대|라인전|대결|싸우|싸울|맞딜)", t),
                re.search(m + r"\s*(으로|로)\s", t),
                t.count(me) >= 2]):
            return True
    return False


def _duel_champs(rng, champs):
    pool = [x for x in champs if x not in DUEL_AVOID and x not in HARD_AVOID]
    return rng.sample(pool, 2)


def duel_frame(rng, A, B, intent, lane):
    f = dict(EMPTY, 의도=["상대법", "템트리"] if intent is None else [intent],
             챔피언=[{"이름": A, "역할": "맞대결"}, {"이름": B, "역할": "맞대결"}])
    f["라인"] = lane
    return f


def write_duel(rng, a, i, fo, why, pair):
    """맞대결 틀 하나 (pair=True면 같은 두 챔피언의 나·상대 표시 문장도 같이) -> 통과 줄 수"""
    A, B = _duel_champs(rng, CHAMPS)
    intent = rng.choice([None, "템트리", "상대법"])
    forms = DUEL_FORMS[intent]
    form = rng.choice(forms)
    lane = rng.choice(LANES) if "{L}" in form else None
    sa, sb = surface_champ(rng, A), surface_champ(rng, B)
    f = duel_frame(rng, A, B, intent, lane)
    ex = [x.format(A=sa, B=sb, L=lane or "") .replace("  ", " ").strip() for x in rng.sample(forms, 3)
          if ("{L}" in x) == bool(lane)][:2] or [form.format(A=sa, B=sb, L=lane or "")]
    ask = {None: "무엇을 물을지 적지 않고 두 챔피언 이름만 맞붙여 씀",
           "템트리": "두 챔피언이 붙을 때 템트리(뭐 살지)를 물음", "상대법": "두 챔피언 중 누가 이기는지·상성·라인전을 물음"}[intent]
    lines = [f"두 챔피언 '{sa}'와 '{sb}'의 맞대결에 대한 아주 짧은 말 (누가 내 챔피언이고 누가 상대인지는 밝히지 않음)",
             ask, "꼴 예: " + ", ".join(f"'{x}'" for x in ex),
             "'나', '내가', '상대', '적', '~로', '~랑 맞라인', '~나왔는데' 같은 말은 쓰지 말 것. 같은 이름을 두 번 쓰지 말 것"]
    if lane:
        lines.append(f"라인 '{lane}'를 꼭 넣을 것")
    else:
        lines.append("라인 이름은 쓰지 말 것")
    meta = {"names": {A: sa, B: sb}, "topic": None, "lines": lines, "must": [sa, sb], "duel": True,
            "same_intent": intent is None}             # 의도를 안 적은 꼴은 의도 단서어 검사 생략
    jobs = [(f, meta, None)]
    if pair:                                          # 대조: 같은 두 챔피언, 같은 의도, 나·상대 표시 있음
        me_first = rng.random() < 0.5
        M, E = (A, B) if rng.random() < 0.5 else (B, A)
        sm, se = meta["names"][M], meta["names"][E]
        vform = rng.choice([x for x, mf in VS_FORMS if mf == me_first])
        f2 = dict(f, 챔피언=[{"이름": M, "역할": "나"}, {"이름": E, "역할": "상대"}])
        tail = {None: "", "템트리": " 템트리", "상대법": " 어케 이김"}[intent]
        ex2 = format_vs(vform, sm, se) + (f" {lane}" if lane else "") + tail
        vform_b = rng.choice([x for x, mf in VS_FORMS if mf == me_first and x != vform])   # 코드 문장 2개 (같은 순서, 다른 꼴)
        ex2b = format_vs(vform_b, sm, se) + (f" {lane}" if lane else "") + tail
        lines2 = [f"질문자는 '{sm}'를 하고 상대(맞라인)는 '{se}'", ask.replace("두 챔피언 이름만 맞붙여 씀", "나와 상대만 밝힘"),
                  f"꼴 예: '{ex2}' — 누가 나이고 누가 상대인지 꼭 드러낼 것 ('나', '내가', '상대', '적', '~상대로' 등)",
                  (f"라인 '{lane}'를 꼭 넣을 것" if lane else "라인 이름은 쓰지 말 것")]
        meta2 = {"names": {M: sm, E: se}, "topic": None, "lines": lines2, "must": [sm, se],
                 "vs_form": (vform, me_first), "same_intent": intent is None, "pair_of": True}
        jobs.append((f2, meta2, [ex2, ex2b]))
    n = 0
    for j, (fj, mj, code_q) in enumerate(jobs):
        style = ("아주 짧게 (15자 안팎)", 15)
        prompt, must = build_prompt(fj, mj, style, 3)
        try:
            qs = parse_lines(call(prompt, a.seed * 100000 + i * 2 + j))[:3]
        except Exception as e:
            why["호출 실패"] += 1
            print("호출 실패", e, flush=True)
            continue
        qs += code_q or [rng.choice(ex)]                # 코드가 꼴 그대로 쓴 문장 (맞대결 1개, 대조 2개)
        kind = "대조" if mj.get("pair_of") else "맞대결"
        for q in qs:
            bad = check(q, fj, mj, must, style[1])
            if not bad and mj.get("duel") and has_mark(q, A, B):
                bad = "나·상대 표시 있음"
            if not bad and mj.get("pair_of") and not has_mark(q, A, B):
                bad = "나·상대 표시 없음 (맞대결로 바뀜)"
            why[f"{kind}: " + (bad or "통과")] += 1
            if bad and a.show_bad:
                print(f"  버림 [{kind}|{bad}] {q}", flush=True)
            if not bad:
                fo.write(json.dumps({"이전": None, "질문": q, "틀": fj, "말투": f"vs:{kind}"}, ensure_ascii=False) + "\n")
                n += 1
    return n


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--per", type=int, default=4, help="틀 하나당 질문 수")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow", type=float, default=0.3, help="이어지는 질문 비율")
    ap.add_argument("--only-vs", action="store_true", help="나+상대 템트리·상대법 틀만 (보충 생성)")
    ap.add_argument("--only-hard", action="store_true", help="실험 11d 남은 오류 6유형 틀만 (run4)")
    ap.add_argument("--show-bad", action="store_true", help="--only-hard에서 버린 질문 출력 (시범용)")
    ap.add_argument("--duel-frames", type=int, default=0, help="--only-vs: 맞대결 틀 수 (run5)")
    ap.add_argument("--pair-frames", type=int, default=0, help="--only-vs: 맞대결 + 같은 두 챔피언 나·상대 대조 짝 틀 수 (run5)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_dir = ROOT / "data" / "frame11" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"exaone_{a.tag}.jsonl"
    why, n_ok, t0 = Counter(), 0, time.time()
    if a.only_vs and (a.duel_frames or a.pair_frames):      # run5: 맞대결 + 대조 짝
        total = a.duel_frames + a.pair_frames
        with open(out, "a", encoding="utf-8") as fo:
            for i in range(total):
                n_ok += write_duel(rng, a, i, fo, why, pair=i >= a.duel_frames)
                fo.flush()
                if (i + 1) % 25 == 0:
                    print(f"{i + 1}/{total} 틀, 통과 질문 {n_ok}, {(time.time() - t0) / 60:.1f}분", flush=True)
        print(f"완료: {out} 통과 {n_ok}줄, {(time.time() - t0) / 60:.1f}분")
        print("결과:", dict(why.most_common()))
        return
    with open(out, "a", encoding="utf-8") as fo:
        for i in range(a.frames):
            if a.only_hard:
                n_ok += run_hard(rng, a, i, fo, why)
                fo.flush()
                if (i + 1) % 25 == 0:
                    el = time.time() - t0
                    print(f"{i + 1}/{a.frames} 틀, 통과 질문 {n_ok}, {el / 60:.1f}분", flush=True)
                continue
            f, meta = sample_frame(rng, CHAMPS, a.only_vs)
            style = pick_style(rng, meta)
            prompt, must = build_prompt(f, meta, style, a.per)
            try:
                qs = parse_lines(call(prompt, a.seed * 100000 + i))[:a.per]
            except Exception as e:
                why["호출 실패"] += 1
                print("호출 실패", e, flush=True)
                continue
            good = []
            for q in qs:
                bad = check(q, f, meta, must, style[1])
                why[bad or "통과"] += 1
                if not bad:
                    good.append(q)
            for q in good:
                fo.write(json.dumps({"이전": None, "질문": q, "틀": f, "말투": style[0]}, ensure_ascii=False) + "\n")
                n_ok += 1
            # 개요는 '챔피언 이름만' 친 질문도 일부 (실제 질문에 많음, 코드가 직접 씀)
            if f["의도"] == ["개요"] and len(f["챔피언"]) == 1 and rng.random() < 0.4:
                q = meta["names"][f["챔피언"][0]["이름"]] + rng.choice(["", "", "?"])
                fo.write(json.dumps({"이전": None, "질문": q, "틀": f, "말투": "이름만"}, ensure_ascii=False) + "\n")
                why["이름만"] += 1
                n_ok += 1
            # 이어지는 질문: 챔피언이 있는 롤 틀이면 일부를 이어받기
            if good and f["범위"] == "롤" and f["챔피언"] and rng.random() < a.follow * 2:
                pq = rng.choice(good)
                f2, meta2 = follow_frame(rng, f, CHAMPS)
                style2 = pick_style(rng, meta2)
                prompt2, must2 = build_prompt(f2, meta2, style2, 2, prev_q=pq)
                try:
                    qs2 = parse_lines(call(prompt2, a.seed * 100000 + i + 50000))[:2]
                except Exception:
                    qs2 = []
                for q in qs2:
                    bad = check(q, f2, meta2, must2, style2[1])
                    if not bad and any(meta["names"][n] in q or n in S.normalize(q) for n in meta2["prev_names"]
                                       if n in meta2["inherit"] or n not in {c["이름"] for c in f2["챔피언"]}):
                        bad = "이어받을 이름을 다시 씀"
                    why[("이어짐 " + bad) if bad else "이어짐 통과"] += 1
                    if not bad:
                        fo.write(json.dumps({"이전": {"질문": pq, "틀": f}, "질문": q, "틀": f2, "말투": "이어짐"},
                                            ensure_ascii=False) + "\n")
                        n_ok += 1
            fo.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{a.frames} 틀, 통과 질문 {n_ok}, {el / 60:.1f}분 (남은 약 {el / (i + 1) * (a.frames - i - 1) / 60:.0f}분)",
                      flush=True)
    print(f"완료: {out} 통과 {n_ok}줄, {(time.time() - t0) / 60:.1f}분")
    print("결과:", dict(why.most_common()))


if __name__ == "__main__":
    main()
