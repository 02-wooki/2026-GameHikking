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

STYLES = [
    "아주 짧은 반말 (5~10글자), 물음표 없어도 됨",
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
DATAOUT = ["두 스킬을 동시에 쓰는 콤보·상호작용 가능 여부(예: 플래시와 스킬)", "룬 세팅", "앞으로의 패치 예측(언제 상향해줘)",
           "스킨 추천·가격", "챔피언 대사·성우", "프로 대회 픽·밴", "특정 구간의 피해량 계산", "정글 동선·와드 위치",
           "랭크 티어 올리는 일반 팁", "챔피언 출시일·역사"]

INTENT_W = {"개요": 7, "스킬": 11, "패치": 14, "이유": 7, "능력치": 6, "템트리": 13, "상대법": 10, "성능": 10,
            "아이템": 7, "추천": 10, "안내": 4}
PAIRS = [("성능", "패치"), ("상대법", "템트리"), ("패치", "이유"), ("스킬", "패치"), ("개요", "템트리"),
         ("템트리", "아이템"), ("개요", "스킬"), ("성능", "템트리"), ("상대법", "스킬"), ("추천", "템트리"),
         ("능력치", "패치"), ("패치", "성능")]


def surface_champ(rng, n):
    return rng.choice(ALIAS_OF[n]) if n in ALIAS_OF and rng.random() < 0.4 else n


def surface_item(rng, n):
    return rng.choice(ITEM_ALIAS_OF[n]) if n in ITEM_ALIAS_OF and rng.random() < 0.5 else n


# ---------------- 1. 틀 뽑기 ----------------
def sample_frame(rng, champs):
    """-> (틀, 말해야 할 것들 meta). meta: 이름 표기, 프롬프트 설명용 정보"""
    f = dict(EMPTY, 의도=[], 챔피언=[])
    meta = {"names": {}, "topic": None}
    r = rng.random()
    if r < 0.15:
        f["범위"] = "사이트밖"
        meta["topic"] = rng.choice(OFFSITE)
        return f, meta
    if r < 0.25:
        f["범위"] = "데이터밖"
        meta["topic"] = rng.choice(DATAOUT)
        if rng.random() < 0.7:
            c = rng.choice(champs)
            f["챔피언"] = [{"이름": c, "역할": "대상"}]
            meta["names"][c] = surface_champ(rng, c)
            if rng.random() < 0.3:
                f["스킬"] = rng.choice(["Q", "W", "E", "R"])
        return f, meta
    if rng.random() < 0.2:
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
        if "상대법" in I or rng.random() < 0.45:
            if rng.random() < 0.8:
                champ("나")
            champ("상대")
        else:
            champ("나")
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
            f["방향"] = rng.choices(["상향", "하향", "변경", None], weights=[3, 3, 1, 4])[0]
            if f["시점"] == "이력" and rng.random() < 0.5:
                f["개수"] = "전체"
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
    meta = {"names": {}, "topic": None, "inherit": [c["이름"] for c in prev["챔피언"]]}
    f["이어받음"] = True
    has_me = any(c["역할"] == "나" for c in f["챔피언"])
    kind = rng.random()
    if has_me and kind < 0.35 and not any(c["역할"] == "상대" for c in f["챔피언"]):
        f["의도"] = ["템트리"] if rng.random() < 0.6 else ["상대법"]
        c = rng.choice([x for x in champs if x not in meta["inherit"]])
        f["챔피언"].append({"이름": c, "역할": "상대"})
        meta["names"][c] = surface_champ(rng, c)
        f["라인"] = prev["라인"]
        meta["mode"] = "상대 추가"
    elif kind < 0.6:
        f["의도"] = ["스킬"]
        f["스킬"] = rng.choice(SKILLS)
        f["챔피언"] = [{"이름": c["이름"], "역할": "대상"} for c in f["챔피언"][:1]]
        meta["mode"] = "스킬로 바꿈"
    elif kind < 0.8:
        f["의도"] = [rng.choice(["패치", "성능", "능력치", "개요"])]
        f["챔피언"] = [{"이름": c["이름"], "역할": "대상"} for c in f["챔피언"][:1]]
        if f["의도"] == ["패치"]:
            f["시점"] = "최신"
        if f["의도"] == ["성능"]:
            f["시점"] = "최신"
        meta["mode"] = "의도 바꿈"
    else:
        f["의도"] = ["템트리"]
        f["챔피언"] = [{"이름": c["이름"], "역할": "나" if c["역할"] in ("나", "대상") else c["역할"]}
                     for c in f["챔피언"]]
        if rng.random() < 0.5:
            f["라인"] = rng.choice(LANES)
        else:
            f["개수"] = "전체"
        f["라인"] = f["라인"] or prev["라인"]
        meta["mode"] = "템트리로"
    return f, meta


# ---------------- 2. 틀 -> 프롬프트 ----------------
def describe(f, meta):
    nm = meta["names"]
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


def build_prompt(f, meta, style, k, prev_q=None):
    lines, must = describe(f, meta)
    p = ["너는 한국 리그 오브 레전드(롤) 유저다. 롤 정보 사이트 채팅창에 입력할 질문을 쓴다.",
         "실제 유저 질문 말투 예: '블츠 최근에 좋나?', '롤 처음인데 무슨 챔피언 할까', '내가 오른이고 적이 카시오페아야', '아우솔의 스킬이 머야'",
         ""]
    if prev_q:
        p.append(f"앞 질문: \"{prev_q}\"")
    p.append("이번 질문의 내용:")
    p += [f"- {x}" for x in lines]
    if must:
        p.append("- 반드시 이 표기를 그대로 포함: " + ", ".join(f"'{m}'" for m in must))
    p += ["",
          "규칙: 위 내용에 없는 다른 챔피언·아이템 이름, 라인, 상향/하향 여부는 쓰지 않는다. 한국어만 쓴다.",
          f"말투: {style}",
          f"서로 표현이 다른 질문 {k}개를 한 줄에 하나씩, 번호·따옴표·설명 없이 질문만 출력하라."]
    return "\n".join(p), must


def call(prompt, seed, temp=0.95):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "top_p": 0.95, "seed": seed, "num_predict": 300}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read().decode())["response"]


# ---------------- 3. 1차 걸러내기 ----------------
BAD_CHARS = re.compile(r"[一-鿿぀-ヿ]")
LANE_WORDS = re.compile(r"탑|정글|미드|원딜|서폿|서포터|바텀|봇|원거리")
DIR_WORDS = re.compile(r"상향|하향|버프|너프|좋아졌|나빠졌|약해졌|강해졌|쎄졌|세졌")
SKILL_WORDS = re.compile(r"(?<![A-Za-z])[QWER](?![A-Za-z])|궁|패시브|스킬")
CLEAN = re.compile(r"^\s*(?:[-*•]|\d+[.)]?(?=\s))\s*")


NAME_RE = re.compile("|".join(re.escape(n) for n in sorted(S.NAMES, key=len, reverse=True)))


def find_names(q):
    """서비스 사전과 같은 방식(normalize 후 긴 이름 먼저)으로 찾은 챔피언 이름"""
    return set(NAME_RE.findall(S.normalize(q)))


ITEM_RE = re.compile("|".join(re.escape(n) for n in sorted(set(S.ITEM_NAMES) | set(S.ITEM_ALIASES), key=len, reverse=True)))
LOL_WORDS = re.compile(r"롤|리그 ?오브|챔피언|챔프|패치|스킬|아이템|템트리|라인|정글|미드|원딜|서폿|탑|랭크|티어|협곡|칼바람|한타|바론|용|소환사")
# 틀에 있는 내용이 질문에 실제로 드러나는지 (없으면 라벨과 질문이 어긋나므로 버림)
INTENT_CUE = {
    "개요": r"누구|뭐야|뭐임|뭔데|머야|무슨 ?챔|어떤 ?챔|소개|컨셉|콘셉|특징|어떤 ?역할|정체|플레이 ?스타일|뭐 ?하는 ?챔",
    "스킬": r"스킬|(?<![A-Za-z])[QWER](?![A-Za-z])|궁|패시브|기술",
    "패치": r"패치|바뀌|바꼈|바뀠|바뀐|변경|변했|변화|달라|상향|하향|버프|너프|업데이트|조정|수정",
    "이유": r"왜|이유|의도|어째서",
    "능력치": r"능력치|체력|공격력|방어력|마법 ?저항|사거리|이속|이동 ?속도|스탯|마나|공속|공격 ?속도",
    "템트리": r"템|아이템|빌드|코어|뭐 ?사|뭘 ?사|구매",
    "상대법": r"상대|이기|이겨|카운터|vs|VS|잡|대처|라인전|맞라인|대응|파훼|압도|만나",
    "성능": r"좋|OP|op|승률|쎄|세\?|센가|강하|강해|사기|메타|성능|티어|할만|할 ?만|어때|어떰|구리|쓰레기|쓸만|1티어|떡상",
    "아이템": r".",
    "추천": r"추천|할까|뭐 ?하|뭐 ?할|골라|픽|고를|할만한|할 ?만한",
    "안내": r"사이트|여기|뭐 ?하는|뭘 ?물어|무엇을|기능|사용법|어떻게 써",
}
SKILL_CUE = {"Q": r"(?<![A-Za-z])Q(?![A-Za-z])", "W": r"(?<![A-Za-z])W(?![A-Za-z])", "E": r"(?<![A-Za-z])E(?![A-Za-z])",
             "R": r"(?<![A-Za-z])R(?![A-Za-z])|궁", "패시브": r"패시브", "전체": r"스킬"}
FIELD_CUE = {
    "난이도": {"쉬움": r"초보|처음|쉬운|쉽|입문|뉴비|롤린이|시작"},
    "개수": {"하나": r"하나|한 ?명|한 ?개|1개|1명|딱", "전체": r"전부|다 |다\?|전체|모든|싹|모두|총정리|풀"},
    "시점": {"이력": r"예전|옛날|과거|처음|역사|지금까지|그동안|변천|이력|여태|출시|역대|쭉"},
    "방향": {"상향": r"상향|버프|좋아졌|강해졌|쎄졌|세졌|좋게", "하향": r"하향|너프|나빠졌|약해졌|구려졌|안 ?좋아"},
}
CLASS_CUE = {"탱커": r"탱", "전사": r"전사|브루저", "마법사": r"마법사|메이지|AP", "암살자": r"암살자|어쌔|암살",
             "원딜": r"원딜|원거리", "서포터": r"서포터|서폿|서포트"}
LANE_CUE = {"탑": r"탑", "정글": r"정글", "미드": r"미드", "원딜": r"원딜|바텀|봇", "서폿": r"서폿|서포터|서포트"}


def check(q, f, meta, must):
    if not q or len(q) < 2 or len(q) > 120:
        return "길이"
    if BAD_CHARS.search(q):
        return "한자·일본어"
    if f["범위"] == "사이트밖":
        return "롤 얘기 섞임" if (find_names(q) or LOL_WORDS.search(q)) else None
    for m in must:
        if m not in q:
            return "표기 빠짐"
    items = set(ITEM_RE.findall(q))
    want_items = {meta["names"][f["아이템"]], f["아이템"]} if f["아이템"] else set()
    if items - want_items and not any(i in w for i in items for w in want_items):
        return "다른 아이템"
    if f["범위"] == "롤":
        for i in f["의도"]:
            if not re.search(INTENT_CUE[i], q):
                return f"의도 '{i}' 안 드러남"
        if f["역할군"] and not re.search(CLASS_CUE[f["역할군"]], q):
            return "역할군 안 드러남"
        for k, cues in FIELD_CUE.items():
            if f[k] in cues and not re.search(cues[f[k]], q):
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


def parse_lines(text):
    out = []
    for ln in text.splitlines():
        ln = CLEAN.sub("", ln).replace("'", "").replace('"', "").strip()
        if ln:
            out.append(ln)
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--per", type=int, default=4, help="틀 하나당 질문 수")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow", type=float, default=0.22, help="이어지는 질문 비율")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_dir = ROOT / "data" / "frame11" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"exaone_{a.tag}.jsonl"
    why, n_ok, t0 = Counter(), 0, time.time()
    with open(out, "a", encoding="utf-8") as fo:
        for i in range(a.frames):
            style = rng.choice(STYLES)
            f, meta = sample_frame(rng, CHAMPS)
            prompt, must = build_prompt(f, meta, style, a.per)
            try:
                qs = parse_lines(call(prompt, a.seed * 100000 + i))[:a.per]
            except Exception as e:
                why["호출 실패"] += 1
                print("호출 실패", e, flush=True)
                continue
            good = []
            for q in qs:
                bad = check(q, f, meta, must)
                why[bad or "통과"] += 1
                if not bad:
                    good.append(q)
            for q in good:
                fo.write(json.dumps({"이전": None, "질문": q, "틀": f, "말투": style}, ensure_ascii=False) + "\n")
                n_ok += 1
            # 이어지는 질문: 챔피언이 있는 롤 틀이면 일부를 이어받기
            if good and f["범위"] == "롤" and f["챔피언"] and rng.random() < a.follow * 2:
                pq = rng.choice(good)
                f2, meta2 = follow_frame(rng, f, CHAMPS)
                prompt2, must2 = build_prompt(f2, meta2, rng.choice(STYLES), 2, prev_q=pq)
                try:
                    qs2 = parse_lines(call(prompt2, a.seed * 100000 + i + 50000))[:2]
                except Exception:
                    qs2 = []
                for q in qs2:
                    bad = check(q, f2, meta2, must2)
                    if not bad and any(meta["names"][n] in q or n in S.normalize(q) for n in meta2["inherit"]):
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
