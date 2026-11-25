# src/gpt/experiments/train_intent10b.py
"""실험 10b: 실험 10과 모든 조건 동일, 모델 입력에서 이름을 자리표시로 바꾸는 것만 추가
실험 10 진단: 학습 말투 100%인데 처음 보는 말투 51.7% (규칙 84.1%). '코르키 소개해줘'를 아이템으로 분류하는 등
  처음 보는 챔피언 이름을 아이템 이름처럼 받아들임 (실험 7~9의 '이름 구분' 약점과 같은 원인)
변경: 사전으로 찾은 챔피언·아이템·스킬 이름과 패치 번호를 [챔피언]/[아이템]/[스킬]/[패치]로 바꾼 뒤 모델에 넣는다
  -> 모델은 이름이 아니라 말투만 보고 의도를 고른다 (이름 찾기는 원래도 사전 담당)

[실험 10 설명]
실험 10: MiniGPT 의도 분류 (질문 -> 의도 하나)

지금 서비스는 질문 의도를 키워드 규칙(search.parse)으로 정한다. 새 말투가 나올 때마다 사람이 키워드를 추가해야 한다.
-> 사전학습 모델(pretrain_v4)을 '질문 -> 의도 번호' 과제로 미세조정해, 처음 보는 말투에도 의도를 맞히는지 본다.
챔피언 이름·패치 번호·슬롯은 지금처럼 사전(규칙)으로 찾는다 (정확히 일치해야 하는 정보라 사전이 더 확실).

의도 9가지 (출력 = 숫자 토큰 하나):
  0 overview 챔피언 개요  1 skill 스킬  2 patch 패치 변경  3 why 변경 이유  4 stats 기본 능력치
  5 build 템트리  6 item 아이템 정보  7 role 역할군 챔피언 목록  8 other 그 밖(잡담·막 친 글자·범위 밖)

데이터: 말투 틀(템플릿) x 챔피언·아이템·패치 등을 채워 생성. 실제 사용자 질문은 아직 없음(로거로 수집 중).
평가 (학습에 안 쓴 것만):
  A. 처음 보는 말투: 의도마다 틀의 약 25%를 학습에서 빼 두고 그 틀로만 만든 문항 (+ 홀드아웃 챔피언 20명)
  B. 손으로 쓴 문항: 사람 말투를 흉내 내 직접 쓴 질문 (틀과 별개). 실제 사용자 말투의 대용이라 한계가 있음
  두 세트 모두 지금 규칙(search.parse)의 정확도를 기준선으로 함께 잰다.
  ※ 평가 세트로 best를 고르지 않고 마지막 회차를 쓴다. B는 규칙을 다듬은 사람(=같은 말투 감각)이 써서 규칙에 유리할 수 있다.

실행 (프로젝트 루트):
  caffeinate -dims /usr/local/bin/python3.10 -u src/gpt/experiments/train_intent10.py
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "gpt"))
sys.path.insert(0, str(ROOT / "src" / "rag"))

import torch
import torch.nn.functional as F

import search as S
import recommend as R
from lol_tokenizer import load as load_tokenizer, format_chat
from mini_gpt import MiniGPT

INIT = ROOT / "checkpoints" / "minigpt_pretrain_v4.pt"
OUT_CKPT = ROOT / "checkpoints" / "minigpt_intent10b.pt"
EVAL_JSON = ROOT / "checkpoints" / "intent10b_eval.json"
BLOCK_SIZE = 192
D_MODEL, N_LAYER, N_HEAD = 256, 6, 8
DROPOUT = 0.1
BATCH = 32
MAX_ITERS = int(os.environ.get("MAX_ITERS", 3000))
EVAL_INTERVAL = 250
WARMUP = 100
LR, MIN_LR = 3e-4, 3e-5
SEED = 1337
N_TEST_PER_INTENT = 60

LABELS = ["overview", "skill", "patch", "why", "stats", "build", "item", "role", "other"]
LID = {l: i for i, l in enumerate(LABELS)}

# ---------------- 채울 값 ----------------
sft_val = (ROOT / "data" / "sft" / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
VAL_CHAMPS = sorted({json.loads(l)["champion"] for l in sft_val if l.strip()} & set(S.CHAMPS))
TRAIN_CHAMPS = sorted(set(S.CHAMPS) - set(VAL_CHAMPS))
ALIAS_OF = defaultdict(list)
for a, n in S.ALIASES.items():
    ALIAS_OF[n].append(a)
PATCHES = sorted(S.ALL_PATCHES)
ITEMS = [n for n in S.ITEM_NAMES if len(n) >= 3]
ITEM_ALIAS = list(S.ITEM_ALIASES)
STATS = ["체력", "마나", "공격력", "공격 속도", "공속", "방어력", "마법 저항력", "마저", "이동 속도", "이속", "사거리", "기본 스탯", "능력치"]
LANES = ["탑", "정글", "미드", "원딜", "바텀", "서폿"]
ROLES = ["탱커", "전사", "브루저", "마법사", "메이지", "암살자", "어쌔신", "원거리 딜러", "서포터"]
SLOT_WORDS = ["Q", "W", "E", "R", "q", "w", "e", "r", "궁", "궁극기", "패시브", "기본 지속 효과"]
CHANGE_FIELDS = ["재사용 대기시간", "쿨", "쿨타임", "피해량", "데미지", "딜", "마나 소모량", "사거리", "보호막", "지속시간", "계수", "이동 속도"]


def champ(rng, pool):
    c = rng.choice(pool)
    r = rng.random()
    if ALIAS_OF.get(c) and r < 0.35:
        return rng.choice(ALIAS_OF[c])
    if " " in c and r < 0.5:
        return c.replace(" ", "")
    return c


def patch_txt(rng):
    p = rng.choice(PATCHES)
    lab = S.patch_label(p)
    return rng.choice([lab, lab, p, lab + "패치", lab + " 패치"])


def skill_name(rng, c):
    names = list(S.SKILL_SLOTS.get(c, {}))
    return rng.choice(names) if names else rng.choice(SLOT_WORDS)


# 의도별 말투 틀. 끝의 True = 평가 A 전용(학습에 안 씀)
T = {
    "overview": [("{c}", 0), ("{c} 알려줘", 0), ("{c} 어떤 챔피언이야", 0), ("{c}에 대해 설명해줘", 0), ("{c} 정보", 0),
                 ("{c} 챔프 설명 좀", 0), ("{c} 스킬 전부 알려줘", 0), ("{c} 어때", 0), ("{c} 뭐하는 챔피언임", 0),
                 ("{c} {l}", 0), ("{c} {l} 어때", 0), ("{c} 소개해줘", 1), ("{c} 처음 해보는데 어떤 챔프임", 1),
                 ("{c} 알려주세요", 1), ("{c} 총정리", 1)],
    "skill": [("{c} {s} 뭐야", 0), ("{c} {s} 스킬 알려줘", 0), ("{c}의 {s} 설명해줘", 0), ("{c} {s} 어떤 스킬임", 0),
              ("{c} {s} 쿨 몇초야", 0), ("{c} {sk} 뭐임", 0), ("{sk} 설명", 0), ("{c} {s} 효과", 0),
              ("{c} {s} 마나 얼마 들어", 0), ("{c} {sk} 어떻게 써", 0), ("{c} {s} 무슨 효과야", 1),
              ("{c} {s}가 뭐였지", 1), ("{sk} 쿨타임", 1), ("{c} {s} 스킬 설명 부탁", 1)],
    "patch": [("{c} {p} 패치 변경점", 0), ("{c} 최근 패치 뭐 바뀜", 0), ("{c} {s} 너프됐어?", 0), ("{c} {sk} {f} 줄었어?", 0),
              ("{c} 버프 먹었냐", 0), ("{p} {c} 바뀐거", 0), ("{c} {s} {f} 올랐음?", 0), ("{c} 이번 패치 상향이야 하향이야", 0),
              ("{c} {p} {s} {f} 어떻게 바뀌었어", 0), ("{c} 패치노트", 0), ("{c} 하향 내용", 0), ("{c} {s} 변경 사항", 1),
              ("{c} {p}에 뭐 달라졌어", 1), ("{c} {sk} 버프됨?", 1), ("{c} 요즘 너프 당했어?", 1)],
    "why": [("{c} 왜 너프됐어", 0), ("{c} {p} 패치 왜 바뀐 거야", 0), ("{c} 너프 이유", 0), ("{c} 버프 왜 해줌", 0),
            ("{c} {s} 왜 바꿨어", 0), ("{c} 하향 의도가 뭐야", 0), ("{c} 왜 상향함", 0), ("{p} {c} 변경 이유", 0),
            ("{c} 너프 왜 당했냐", 1), ("{c} 패치 의도 알려줘", 1), ("{c} {s} 왜 너프함", 1)],
    "stats": [("{c} {st} 얼마야", 0), ("{c} 기본 {st}", 0), ("{c} {st} 몇이야", 0), ("{c}의 {st} 알려줘", 0),
              ("{c} {st} 수치", 0), ("{c} 기본 스탯 알려줘", 0), ("{c} {st} 높아?", 0), ("{c} 레벨당 {st}", 0),
              ("{c} {st}이 어떻게 돼", 1), ("{c} {st} 몇임", 1), ("{c} 스탯 좀", 1)],
    "build": [("{c} 템트리", 0), ("{c} {l} 템트리 추천해줘", 0), ("{c} 뭐 사야 돼", 0), ("{c} 코어템 뭐야", 0),
              ("상대 {e} 나 {c} 템트리", 0), ("{c} vs {e} 뭐 사", 0), ("{c} {l} 빌드", 0), ("{c} 아이템 추천", 0),
              ("{c}로 {e} 상대할 때 템", 0), ("{c} 첫 코어 뭐감", 0), ("{c} {l} 템 뭐 가", 0), ("{c} 신발 뭐 신어", 0),
              ("{e} 상대로 {c} 템 어떻게 가", 1), ("{c} 무슨 템 가야함", 1), ("{c} {l} 아이템 순서", 1),
              ("나 {c}인데 상대 {e}면 뭐 사", 1)],
    "item": [("{it} 가격", 0), ("{it} 얼마야", 0), ("{it} 조합법", 0), ("{it} 재료 뭐야", 0), ("{it} 효과 알려줘", 0),
             ("{it} 능력치", 0), ("{it} 몇 골드", 0), ("{it} 뭐임", 0), ("{it} 상위템", 0), ("{it} 패시브 뭐야", 1),
             ("{it} 만들려면 뭐 필요해", 1), ("{it} 골드 얼마 들어", 1)],
    "role": [("{r} 챔피언 추천해줘", 0), ("{r} 챔프 뭐 있어", 0), ("{r} 목록", 0), ("{r} 챔피언 알려줘", 0),
             ("{r} 할만한 거", 0), ("{r} 챔프 리스트", 0), ("{r}는 누가 있어", 1), ("{r} 챔피언 종류", 1)],
    "other": [("안녕", 0), ("테스트", 0), ("ㅋㅋㅋㅋ", 0), ("롤 재밌다", 0), ("오늘 날씨 어때", 0), ("너 누구야", 0),
              ("배고프다", 0), ("ㅁㄴㅇㄹ", 0), ("asdf", 0), ("티어 올리는 법", 0), ("롤 접을까", 0), ("고마워", 0),
              ("뭐해", 1), ("이거 어떻게 써", 1), ("ㅎㅇ", 1), ("ㄴㅇㄹㄴㅇㄹ", 1), ("점심 뭐 먹지", 1)],
}
TAILS = ["", "", "", "?", "??", " 좀", "요", " ㅠㅠ", " ㅋㅋ", "!", " 알려줘", " 궁금함"]


def fill(tpl, rng, pool):
    c = champ(rng, pool)
    canon = S.normalize(c) if c not in S.CHAMPS else c
    e = champ(rng, [x for x in pool if x != canon] or pool)
    q = tpl.format(c=c, e=e, s=rng.choice(SLOT_WORDS), sk=skill_name(rng, canon), p=patch_txt(rng),
                   f=rng.choice(CHANGE_FIELDS), st=rng.choice(STATS), l=rng.choice(LANES),
                   it=rng.choice(ITEMS + ITEM_ALIAS), r=rng.choice(ROLES))
    if not q.endswith(("?", "ㅋ", "ㅠ")) and rng.random() < 0.5:
        q += rng.choice(TAILS)
    if rng.random() < 0.15:
        q = q.replace(" ", "")                         # 띄어쓰기 생략
    return q.strip()


def gen(intent, rng, test, pool):
    tpls = [t for t, held in T[intent] if bool(held) == test]
    return fill(rng.choice(tpls), rng, pool)


# ---------------- B. 손으로 쓴 문항 (틀과 별개, 사람 말투 흉내) ----------------
HAND = [
    ("야스오", "overview"), ("요네 알려줘봐", "overview"), ("럭스 어떤 챔이야", "overview"), ("제드 처음인데 뭐하는 챔프임", "overview"),
    ("마이 정보 좀", "overview"), ("블츠 서폿", "overview"), ("아리 미드 어떰", "overview"), ("케틀 원딜 괜찮음?", "overview"),
    ("리신 정글", "overview"), ("말파 탑 해볼라는데", "overview"),
    ("아리 큐 뭐임", "skill"), ("럭스 궁 사거리 얼마나 됨", "skill"), ("야스오 w 바람장막 효과", "skill"), ("제드 궁 쿨 몇초", "skill"),
    ("리신 패시브 뭐였더라", "skill"), ("이렐 e 어케 씀", "skill"), ("카타 r 설명좀", "skill"), ("누누 q 마나 얼마", "skill"),
    ("블츠 그랩 쿨", "skill"), ("징크스 패시브 뭐임", "skill"),
    ("아트록스 요즘 너프먹음?", "patch"), ("제드 이번패치 버프임?", "patch"), ("럭스 궁 딜 줄었냐", "patch"),
    ("마이 26.19 뭐바뀜", "patch"), ("야스오 패치 내역", "patch"), ("리신 q 쿨 늘었어?", "patch"), ("케틀 상향됐다던데", "patch"),
    ("아리 최근에 바뀐거 있음", "patch"), ("다리우스 너프 언제 됐어", "patch"), ("카이사 26.20 변경", "patch"),
    ("아트록스 왜 너프함", "why"), ("제드 버프 이유가 뭐임", "why"), ("야스오 너프 왜 당한거임", "why"),
    ("럭스 궁 왜 줄였대", "why"), ("리신 하향한 이유", "why"), ("케틀 왜 상향해줌?", "why"),
    ("아리 체력 몇임", "stats"), ("다리우스 공격력 얼마", "stats"), ("마이 이속 몇", "stats"), ("제드 방어력", "stats"),
    ("누누 기본 체력 높음?", "stats"), ("케틀 사거리 몇이야", "stats"), ("야스오 기본스탯", "stats"),
    ("나서스 템 뭐감", "build"), ("마이 탑 뭐 사야됨", "build"), ("상대 다리우스인데 가렌 템 어케 감", "build"),
    ("야스오 코어템", "build"), ("럭스 서폿 빌드 추천", "build"), ("제드 vs 아리 템트리", "build"), ("리신 첫템 뭐감", "build"),
    ("케틀 신발 뭐 신음", "build"), ("아리 템 추천좀", "build"), ("말파 상대 이렐 뭐삼", "build"),
    ("몰왕 얼마임", "item"), ("삼위일체 재료", "item"), ("존야 효과 뭐임", "item"), ("무대 가격", "item"),
    ("가갑 몇골", "item"), ("라바돈 능력치", "item"), ("구인수 조합 어케함", "item"),
    ("탱커 챔프 추천", "role"), ("암살자 뭐있음", "role"), ("원딜 챔 목록", "role"), ("서폿 할만한거 추천", "role"),
    ("메이지 챔피언 뭐 있어", "role"), ("브루저 추천좀", "role"),
    ("ㅇㅇ", "other"), ("테스트입니다", "other"), ("롤 너무 어렵다", "other"), ("ㅁ닝ㅁㄴㅇ러ㅣ", "other"),
    ("이거 뭐하는 사이트임", "other"), ("다이아 가고싶다", "other"), ("ㅋㅋㅋ 뭐야", "other"), ("hi", "other"),
]


def rule_label(q):
    P = S.parse(q)
    if P.intent == "patch":
        return "why" if P.why else "patch"
    if P.intent == "free":
        return "other"
    return P.intent if P.intent in LID else "other"


# ---------------- 모델 ----------------
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB, PAD_ID = len(tok), tok.pad_token_id
END_ID = tok.convert_tokens_to_ids("<|end|>")
DIGIT = []
for i in range(len(LABELS)):
    ids = tok.encode(f"{i}<|end|>", add_special_tokens=False)
    assert len(ids) == 2 and ids[1] == END_ID
    DIGIT.append(ids[0])


import re
_TERMS = {}
for _n in S.NAMES:
    _TERMS[_n] = "[챔피언]"
for _n in list(S.ITEM_NAMES) + list(S.ITEM_ALIASES):
    _TERMS.setdefault(_n, "[아이템]")
for _m in S.SKILL_SLOTS.values():
    for _n in _m:
        if len(_n) >= 2:
            _TERMS.setdefault(_n, "[스킬]")
_TERM_RE = re.compile("|".join(re.escape(t) for t in sorted(_TERMS, key=len, reverse=True)))


def mask(q):
    """이름은 사전 담당: 챔피언·아이템·스킬 이름, 패치 번호를 자리표시로 바꾼다"""
    q = S.normalize(q)
    q = S.PATCH_RE.sub("[패치]", q)
    return _TERM_RE.sub(lambda m: _TERMS[m[0]], q)


def enc(q):
    p = format_chat([{"role": "system", "content": "질문 의도 분류"}, {"role": "user", "content": mask(q)}],
                    add_generation_prompt=True)
    return tok.encode(p, add_special_tokens=False)[-BLOCK_SIZE:]


def make_set(rng, n_per, test, pool):
    return [(gen(l, rng, test, pool), l) for l in LABELS for _ in range(n_per)]


rng_tr = random.Random(SEED)
test_A = make_set(random.Random(7), N_TEST_PER_INTENT, True, VAL_CHAMPS)
diag_tr = make_set(random.Random(8), 30, False, TRAIN_CHAMPS)
print(f"디바이스 {device} | 평가 A {len(test_A)}문항 (처음 보는 말투 + 홀드아웃 챔피언) | 평가 B {len(HAND)}문항 (손으로 씀)")
print("평가 A 예시:", [q for q, _ in random.Random(1).sample(test_A, 6)])
print("모델 입력 예시:", [mask(q) for q, _ in random.Random(1).sample(test_A, 6)] + [mask(q) for q, _ in HAND[:4]])


def rule_acc(items):
    return sum(rule_label(q) == l for q, l in items) / len(items)


RULE_A, RULE_B = rule_acc(test_A), rule_acc(HAND)
print(f"규칙 기준선: 평가 A {RULE_A * 100:.1f}% | 평가 B {RULE_B * 100:.1f}%")

model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD,
                block_size=BLOCK_SIZE, dropout=DROPOUT).to(device)
ck0 = torch.load(INIT, map_location=device)
model.load_state_dict(ck0["model"])
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
digit_t = torch.tensor(DIGIT, device=device)


def logits_of(qs):
    ps = [enc(q) for q in qs]
    L = max(len(p) for p in ps)
    x = torch.full((len(ps), L), PAD_ID, dtype=torch.long)
    for j, p in enumerate(ps):
        x[j, :len(p)] = torch.tensor(p)
    x = x.to(device)
    last = torch.tensor([len(p) - 1 for p in ps], device=device)
    return model(x)[torch.arange(len(ps), device=device), last][:, digit_t].float()   # 9개 숫자 토큰만


@torch.no_grad()
def evaluate(items, keep=0):
    model.eval()
    preds, conf = [], []
    for i in range(0, len(items), 64):
        pr = F.softmax(logits_of([q for q, _ in items[i:i + 64]]), -1)
        c, p = pr.max(-1)
        preds += p.tolist()
        conf += c.tolist()
    model.train()
    ok = [LABELS[p] == l for p, (_, l) in zip(preds, items)]
    by = defaultdict(lambda: [0, 0])
    for o, (_, l) in zip(ok, items):
        by[l][0] += o
        by[l][1] += 1
    fails = [{"q": q, "gold": l, "pred": LABELS[p], "conf": round(c, 2)}
             for (q, l), p, c, o in zip(items, preds, conf, ok) if not o][:keep]
    return sum(ok) / len(ok), {k: f"{a}/{b}" for k, (a, b) in by.items()}, fails


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


best, running, t0 = -1.0, 0.0, time.time()
model.train()
for it in range(1, MAX_ITERS + 1):
    for g in opt.param_groups:
        g["lr"] = get_lr(it - 1)
    batch = [(gen(l, rng_tr, False, TRAIN_CHAMPS), l) for l in rng_tr.choices(LABELS, k=BATCH)]
    loss = F.cross_entropy(logits_of([q for q, _ in batch]), torch.tensor([LID[l] for _, l in batch], device=device))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        a, _, _ = evaluate(test_A)
        b, _, _ = evaluate(HAND)
        tr, _, _ = evaluate(diag_tr)
        mark = ""
        if it == MAX_ITERS:                # 평가 세트로 체크포인트를 고르면 점수가 부풀므로 마지막 회차를 그대로 쓴다
            torch.save({"model": model.state_dict(), "iter": it, "labels": LABELS,
                        "config": dict(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD,
                                       block_size=BLOCK_SIZE)}, OUT_CKPT)
            mark = "  ← 저장 (마지막 회차)"
        print(f"[{it}/{MAX_ITERS}] 손실 {running / EVAL_INTERVAL:.4f} | 학습 말투 {tr * 100:5.1f}% | "
              f"A 처음 보는 말투 {a * 100:5.1f}% | B 손으로 쓴 문항 {b * 100:5.1f}% | {time.time() - t0:.0f}초{mark}")
        running = 0.0

ck = torch.load(OUT_CKPT, map_location=device)
model.load_state_dict(ck["model"])
a, a_by, a_f = evaluate(test_A, keep=15)
b, b_by, b_f = evaluate(HAND, keep=30)
rule_b_fail = [{"q": q, "gold": l, "rule": rule_label(q)} for q, l in HAND if rule_label(q) != l]
res = {"best_iter": ck["iter"], "labels": LABELS,
       "A_unseen_phrasing": {"model": a, "rule": RULE_A, "by_intent": a_by, "n": len(test_A)},
       "B_handwritten": {"model": b, "rule": RULE_B, "by_intent": b_by, "n": len(HAND)},
       "model_fails_B": b_f, "rule_fails_B": rule_b_fail, "model_fails_A": a_f}
EVAL_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
