# src/gpt/experiments/train_sft7.py
"""실험 7: 실험 6과 모든 조건 동일, 드롭아웃만 추가 (사전학습 0.2 = pretrain_v4, SFT 0.1)
실험 6 진단: 큰 모델(723만)이 사전학습에서 과적합 -> 나쁜 출발점 -> SFT도 기준 미달 (판별 67.2% / 4지선다 38.7%)
[실험 6 설명] 5e와 모든 조건 동일, 모델 크기만 키움 (d128/L4/H4 -> d256/L6/H8)
5e는 학습 86% / 검증 84%로 과소적합 -> 용량을 늘려 같은 기준(판별 70%, 4지선다 50%)으로 비교

[5d 설명] 실험 5d: 실험 5c와 같은 판별 과제를, 재사전학습 모델(pretrain_v2)에서 시작해 학습
변경: 가중치 초기화 = checkpoints/minigpt_pretrain_v4.pt, 학습률 3e-4 (사전학습 지식 보존)
통과 기준(사전 확정): 검증 판별 70% 이상, 4지선다 50% 이상

[실험 5c 설명]
입력 = 근거 한 줄 + 질문, 출력 = 1(이 줄이 질문의 답) / 0(아님).
서비스에서는 후보 줄마다 P(1)을 계산해 가장 높은 줄을 고른다 (최고값 < 0.5면 '근거 없음').

실험 5/5b(4줄 중 번호 선택)는 학습 데이터에서도 우연 수준이었고, 예측이 번호 하나로 몰렸다.
-> 번호를 옮기는 과정을 없애고 '질문과 한 줄 비교'만 남긴다. 시퀀스도 약 80토큰으로 짧다.
- 학습: 정답 줄 50% / 오답 줄 50% (오답 중 HARD_P는 같은 챔피언 다른 스킬 등 어려운 방해 줄)
- 증강: AUG_P 비율로 무작위 결합 증강 (암기 방지, 홀드아웃 이름 토큰도 학습에 등장)
- 검증: 5/5b와 같은 4지선다 문항 세트(같은 시드)로 '선택 정확도'를 직접 비교 + 규칙 기반 기준선
- 손실은 정답 토큰(0/1) 하나에만 -> 우연 수준 손실 = ln2 = 0.693

실행 (프로젝트 루트에서, 전원 연결 + 덮개 열어 두기):
  caffeinate -dims /usr/local/bin/python3.10 src/gpt/experiments/train_sft7.py
"""
import json
import math
import sys
import random
import re
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import build_sft_data as B
from lol_tokenizer import load as load_tokenizer, format_chat
from mini_gpt import MiniGPT

# ---------------- 설정 ----------------
ROOT = Path(__file__).resolve().parents[3]
INIT = ROOT / "checkpoints" / "minigpt_pretrain_v4.pt"
OUT_CKPT = ROOT / "checkpoints" / "minigpt_sft7.pt"
EVAL_JSON = ROOT / "checkpoints" / "sft7_eval.json"

BLOCK_SIZE = 192
BATCH_SIZE = 32
D_MODEL, N_LAYER, N_HEAD = 256, 6, 8
DROPOUT = 0.1                          # 실험 6: 0 (유일한 변경점, 평가 시에는 model.eval()로 꺼짐)
MAX_ITERS = 15000
EVAL_INTERVAL = 500
WARMUP = 200
LR, MIN_LR = 3e-4, 3e-5
LINE_CHARS = 80                        # 근거 줄마다 보여줄 앞부분 글자 수
AUG_P = 0.5                            # 학습 샘플 중 무작위 결합 증강 비율
HARD_P = 0.5                           # 오답 줄 중 어려운 방해 줄 비율
NEG_P = 0.10                           # (선택 평가용) 정답 줄이 없는 문항 비율
TYPE_W = {"skill": 0.55, "stat": 0.10, "patch": 0.35}
N_EVAL = 300
SEED = 1337

train_rng = random.Random(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 디바이스: {device}")

tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB = len(tok)
PAD_ID = tok.pad_token_id
END_ID = tok.convert_tokens_to_ids("<|end|>")

ANS_IDS = {}
for i in range(5):                     # 0~4 모두 확인 (5/5b 검증 세트 생성과 동일 조건 유지)
    ids = tok.encode(f"{i}<|end|>", add_special_tokens=False)
    assert len(ids) == 2 and ids[1] == END_ID, f"숫자 {i} 토큰화가 예상과 다름: {ids}"
    ANS_IDS[i] = ids[0]
NO_ID, YES_ID = ANS_IDS[0], ANS_IDS[1]

# ---------------- 풀 구성 (train_sft5와 동일) ----------------
base, patch = B.load_base(), B.load_patch()
sft_val = (ROOT / "data" / "sft" / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
val_champs = {json.loads(l)["champion"] for l in sft_val if l.strip()}
print(f"챔피언 {len({b['champ'] for b in base})}명 | 검증 홀드아웃 {len(val_champs)}명")


def build_pools(is_val):
    sk, st = [], []
    for b in base:
        if (b["champ"] in val_champs) != is_val:
            continue
        for s in b["skills"]:
            sk.append((b["champ"], B.base_slot(s["slot"]), s["line"]))
        if all(k in b["stats"] for k in B.STAT_KEYS):
            st.append((b["champ"], b["stats"]))
    keys, cand = defaultdict(int), []
    for p in patch:
        if (p["champ"] in val_champs) != is_val:
            continue
        for l in p["changes"]:
            parts = l.split(": ", 2)
            if len(parts) < 3 or "⇒" not in parts[2]:
                continue
            key = (p["champ"], p["patch"], B.slot_of(l), parts[1])
            keys[key] += 1
            cand.append((p["champ"], p["patch"], B.slot_of(l), l, parts[1], key))
    pt = [c for c in cand if keys[c[5]] == 1]
    return sk, st, pt


train_pools, val_pools = build_pools(False), build_pools(True)
all_names = sorted({b["champ"] for b in base} | {p["champ"] for p in patch})
patch_vers = sorted({p["patch"] for p in patch})
print(f"풀 크기 train(스킬/스탯/패치) {[len(x) for x in train_pools]} | val {[len(x) for x in val_pools]}")


def type_weights(pools):
    w = {}
    for i, t in enumerate(["skill", "stat", "patch"]):
        w[t] = TYPE_W[t] if len({x[0] for x in pools[i]}) >= 5 else 0.0
    return w


TRAIN_W, VAL_W = type_weights(train_pools), type_weights(val_pools)


# ---------------- 실제 데이터 문항 (train_sft5와 동일) ----------------
def sname(slot, rng):
    return rng.choice(B.SLOT_NAMES.get(slot, [slot]))


def stat_line(c, s):
    return f"{c} 기본 능력치 - " + ", ".join(f"{k}: {s[k]}" for k in B.STAT_KEYS)


def fmt_patch(x):
    return f"{x[0]} {x[1]} 패치 - {x[3]}"


def real_skill(P, rng):
    sk = P[0]
    c, slot, line = rng.choice(sk)
    hard = [x for x in sk if x[0] == c and x[1] != slot]
    others = [x for x in rng.sample(sk, min(8, len(sk))) if x[0] != c]
    d = ([rng.choice(hard)] if hard and rng.random() < 0.5 else []) + others
    q = rng.choice(B.SKILL_Q).format(c=c, s=sname(slot, rng))
    return "skill", f"{c} {line}", [f"{x[0]} {x[2]}" for x in d], q


def real_stat(P, rng):
    st = P[1]
    c, s = rng.choice(st)
    key = rng.choice(B.STAT_KEYS)
    others = [x for x in rng.sample(st, min(6, len(st))) if x[0] != c]
    q = rng.choice(B.STAT_Q).format(c=c, k=key)
    return "stat", stat_line(c, s), [stat_line(*x) for x in others], q


def real_patch(P, rng):
    pt = P[2]
    x = rng.choice(pt)
    c, pv, slot, line, field, key = x
    hard = [y for y in pt if y[0] == c and y[1] == pv and y[5] != key]
    others = [y for y in rng.sample(pt, min(8, len(pt))) if y[0] != c]
    d = ([rng.choice(hard)] if hard and rng.random() < 0.5 else []) + others
    q = f"{c} {pv} 패치 {sname(slot, rng)} {field} 어떻게 바뀌었어?"
    return "patch", fmt_patch(x), [fmt_patch(y) for y in d], q


# ---------------- 무작위 결합 증강 (train_sft5와 동일) ----------------
def parse_skills(sk_pool):
    out = []
    for _, _, line in sk_pool:
        m = B.SKILL_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def parse_patch(pt_pool):
    out = []
    for _, _, _, line, _, _ in pt_pool:
        parts = line.split(": ", 2)
        if len(parts) == 3:
            out.append(tuple(parts))
    return out


AUG_SK, AUG_PT = parse_skills(train_pools[0]), parse_patch(train_pools[2])


def other_name(rng, avoid):
    while True:
        n = rng.choice(all_names)
        if n not in avoid:
            return n


def skill_line(name, label, sk, desc):
    return f"{name} {label} - {sk}: {desc}"


def aug_skill(P, rng):
    A = rng.choice(AUG_SK)
    c = rng.choice(all_names)
    tgt = skill_line(c, A[0], A[1], rng.choice(AUG_SK)[2])
    q = rng.choice(B.SKILL_Q).format(c=c, s=sname(B.base_slot(A[0]), rng))
    d = []
    if rng.random() < 0.5:          # 같은 챔피언, 다른 슬롯
        X = rng.choice([p for p in AUG_SK if p[0] != A[0]])
        d.append(skill_line(c, X[0], X[1], rng.choice(AUG_SK)[2]))
    if rng.random() < 0.5:          # 같은 슬롯, 다른 챔피언
        X = rng.choice(AUG_SK)
        d.append(skill_line(other_name(rng, {c}), A[0], X[1], rng.choice(AUG_SK)[2]))
    while len(d) < 4:
        X = rng.choice(AUG_SK)
        d.append(skill_line(other_name(rng, {c}), X[0], X[1], rng.choice(AUG_SK)[2]))
    return "skill", tgt, d, q


def rand_stats(rng):
    return {"기본 체력": rng.randint(480, 720), "공격력": rng.randint(45, 75),
            "방어력": rng.randint(18, 48), "이동속도": rng.randint(325, 355)}


def aug_stat(P, rng):
    c = rng.choice(all_names)
    key = rng.choice(B.STAT_KEYS)
    names = []
    while len(names) < 4:
        names.append(other_name(rng, {c, *names}))
    q = rng.choice(B.STAT_Q).format(c=c, k=key)
    return "stat", stat_line(c, rand_stats(rng)), [stat_line(n, rand_stats(rng)) for n in names], q


def patch_line(name, pv, label, field, vals):
    return f"{name} {pv} 패치 - {label}: {field}: {vals}"


def qkey(p):
    return (B.slot_of(p[0] + ": " + p[1]), p[1])


def aug_patch(P, rng):
    A = rng.choice(AUG_PT)
    c, pv = rng.choice(all_names), rng.choice(patch_vers)
    tgt = patch_line(c, pv, A[0], A[1], rng.choice(AUG_PT)[2])
    q = f"{c} {pv} 패치 {sname(qkey(A)[0], rng)} {A[1]} 어떻게 바뀌었어?"
    d = []
    if rng.random() < 0.5:
        X = rng.choice([p for p in AUG_PT if qkey(p) != qkey(A)])
        d.append(patch_line(c, pv, X[0], X[1], rng.choice(AUG_PT)[2]))
    if rng.random() < 0.5:
        d.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers), A[0], A[1], rng.choice(AUG_PT)[2]))
    while len(d) < 4:
        X = rng.choice(AUG_PT)
        d.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers), X[0], X[1], rng.choice(AUG_PT)[2]))
    return "patch", tgt, d, q


REAL = {"skill": real_skill, "stat": real_stat, "patch": real_patch}
AUG = {"skill": aug_skill, "stat": aug_stat, "patch": aug_patch}


# ---------------- 판별 쌍 (학습·진단용) ----------------
def cut(line):
    return line if len(line) <= LINE_CHARS else line[:LINE_CHARS] + "…"


def enc_pair(q, line):
    prompt = format_chat([{"role": "system", "content": f"근거: {cut(line)}"},
                          {"role": "user", "content": q}], add_generation_prompt=True)
    return tok.encode(prompt, add_special_tokens=False)


drop = {"long": 0, "short": 0}


def make_pair(P, rng, weights, aug):
    typ = rng.choices(list(weights), list(weights.values()))[0]
    kind, tgt, dists, q = (AUG if aug else REAL)[typ](P, rng)
    dists = [d for d in dists if d != tgt]
    if not dists:
        drop["short"] += 1
        return None
    if rng.random() < 0.5:
        line, y = tgt, 1
    else:
        line = dists[0] if rng.random() < HARD_P else rng.choice(dists)   # dists[0]은 어려운 방해 줄일 수 있음
        y = 0
    p = enc_pair(q, line)
    if len(p) + 2 > BLOCK_SIZE:
        drop["long"] += 1
        return None
    a = YES_ID if y else NO_ID
    return {"kind": kind, "y": y, "prompt_ids": p,
            "ids": p + [a, END_ID], "labels": [-100] * len(p) + [a, -100]}   # 손실은 정답 토큰 하나에만


def pair_n(P, rng, n, weights, aug_p):
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 50:
            raise RuntimeError("샘플 생성 실패가 너무 많음")
        s = make_pair(P, rng, weights, rng.random() < aug_p)
        if s:
            out.append(s)
    return out


# ---------------- 선택 평가 세트 (5/5b와 같은 문항이 나오도록 난수 순서 동일) ----------------
def make_set(P, rng, K, weights):
    typ = rng.choices(list(weights), list(weights.values()))[0]
    kind, tgt, dists, q = REAL[typ](P, rng)
    dists = [d for d in dists if d != tgt]
    neg = rng.random() < NEG_P
    need = K if neg else K - 1
    if len(dists) < need:
        return None
    lines = dists[:need] + ([] if neg else [tgt])
    rng.shuffle(lines)
    label = 0 if neg else lines.index(tgt) + 1
    return {"kind": ("neg_" if neg else "") + kind, "q": q, "lines": lines, "label": label}


def set_n(P, rng, n, K, weights):
    out = []
    while len(out) < n:
        rng.random()                  # 5/5b의 sample_n이 증강 여부 판단에 쓰던 난수 (순서 맞춤)
        s = make_set(P, rng, K, weights)
        if s:
            out.append(s)
    return out


val_sel = {K: set_n(val_pools, random.Random(2024 + K), N_EVAL, K, VAL_W) for K in (2, 4)}
val_pairs = pair_n(val_pools, random.Random(99), 600, VAL_W, 0.0)
diag_real = pair_n(train_pools, random.Random(77), 400, TRAIN_W, 0.0)
diag_aug = pair_n(train_pools, random.Random(78), 400, TRAIN_W, 1.0)

ex = val_sel[4][0]
print("\n--- 검증 선택 문항 예시 (K=4, 5/5b와 같아야 함) ---")
print(f"[질문] {ex['q']}  [정답] {ex['label']}")
print("--- 모델 입력 예시 (첫 줄) ---\n" + tok.decode(enc_pair(ex["q"], ex["lines"][0])) + "\n")


# ---------------- 규칙 기반 기준선 ----------------
NAMES_BY_LEN = sorted(all_names, key=len, reverse=True)
WORD_MAP = {"궁극기": "R", "패시브": "기본 지속 효과"}


def rule_pick(q, lines):
    """줄이 질문의 챔피언명으로 시작하고, 질문 단어가 가장 많이 겹치는 줄을 고른다. 없으면 0."""
    champ = next((n for n in NAMES_BY_LEN if q.startswith(n)), None)
    if champ is None:
        return 0
    words = [WORD_MAP.get(w, w) for w in q[len(champ):].split() if w]
    best, pick = -1, 0
    for i, l in enumerate(lines):
        l = cut(l)
        if not l.startswith(champ + " "):
            continue
        s = sum(1 for w in words if w in l)
        if s > best:
            best, pick = s, i + 1
    return pick


for K in (2, 4):
    ok = sum(rule_pick(it["q"], it["lines"]) == it["label"] for it in val_sel[K])
    print(f"규칙 기반 기준선 K={K}: {ok}/{N_EVAL} ({ok / N_EVAL * 100:.1f}%)")


# ---------------- 모델 ----------------
def collate(items):
    L = max(len(it["ids"]) for it in items) - 1
    x = torch.full((len(items), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), L), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["ids"]) - 1
        x[i, :n] = torch.tensor(it["ids"][:-1])
        y[i, :n] = torch.tensor(it["labels"][1:])
    return x.to(device), y.to(device)


model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                n_head=N_HEAD, block_size=BLOCK_SIZE, dropout=DROPOUT).to(device)
ck0 = torch.load(INIT, map_location=device)
model.load_state_dict(ck0["model"])
print(f"\n사전학습 가중치 로드: {INIT.name} (iter {ck0['iter']}, 검증 손실 {ck0['val_loss']:.4f})")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


@torch.no_grad()
def p_match(prompts):
    """각 프롬프트에 대해 P(1) (0/1 두 토큰 중에서의 확률)"""
    out = []
    for i in range(0, len(prompts), 64):
        b = prompts[i:i + 64]
        L = max(len(p) for p in b)
        x = torch.full((len(b), L), PAD_ID, dtype=torch.long)
        for j, p in enumerate(b):
            x[j, :len(p)] = torch.tensor(p)
        x = x.to(device)
        last = torch.tensor([len(p) - 1 for p in b], device=device)
        lg = model(x)[torch.arange(len(b), device=device), last][:, [NO_ID, YES_ID]]
        out += F.softmax(lg.float(), dim=-1)[:, 1].tolist()
    return out


def eval_pairs(pairs):
    model.eval()
    ps = p_match([p["prompt_ids"] for p in pairs])
    model.train()
    acc = sum((p >= 0.5) == bool(s["y"]) for p, s in zip(ps, pairs)) / len(pairs)
    loss = -sum(math.log(max(p if s["y"] else 1 - p, 1e-9)) for p, s in zip(ps, pairs)) / len(pairs)
    return acc, loss


def eval_select(items, keep_fails=0):
    model.eval()
    ps = p_match([enc_pair(it["q"], l) for it in items for l in it["lines"]])
    model.train()
    pos = ok_n = top1_ok = n_pos = 0
    by, dist, fails = defaultdict(lambda: [0, 0]), defaultdict(int), []
    for it in items:
        k = len(it["lines"])
        pi = ps[pos:pos + k]
        pos += k
        bi = max(range(k), key=lambda j: pi[j])
        pred = bi + 1 if pi[bi] >= 0.5 else 0
        ok = pred == it["label"]
        ok_n += ok
        dist[pred] += 1
        by[it["kind"]][0] += ok
        by[it["kind"]][1] += 1
        if it["label"] > 0:
            n_pos += 1
            top1_ok += (bi + 1) == it["label"]
        if not ok and len(fails) < keep_fails:
            fails.append({"q": it["q"], "label": it["label"], "pred": pred,
                          "lines": [(cut(l), round(p, 3)) for l, p in zip(it["lines"], pi)]})
    n = len(items)
    return {"acc": ok_n / n, "top1_pos": top1_ok / max(1, n_pos),
            "by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by.items())},
            "pred_dist": dict(sorted(dist.items())), "fails": fails}


# ---------------- 학습 ----------------
print("🚀 학습 시작 (질문-한 줄 일치 판별)")
print("우연 수준: 판별 정확도 50%, 손실 0.693 | 선택 정확도 K=2 약 33~50%, K=4 약 20~25%")
model.train()
best, running, t0 = -1.0, 0.0, time.time()
for it in range(1, MAX_ITERS + 1):
    for g in opt.param_groups:
        g["lr"] = get_lr(it - 1)
    x, y = collate(pair_n(train_pools, train_rng, BATCH_SIZE, TRAIN_W, AUG_P))
    loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()

    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        va, vl = eval_pairs(val_pairs)
        tr, _ = eval_pairs(diag_real)
        ta, _ = eval_pairs(diag_aug)
        s2, s4 = eval_select(val_sel[2]), eval_select(val_sel[4])
        mark = ""
        if s4["acc"] > best:
            best = s4["acc"]
            OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "iter": it, "val_k4_select_acc": best,
                        "config": dict(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                                       n_head=N_HEAD, block_size=BLOCK_SIZE)}, OUT_CKPT)
            mark = "  ← best 저장"
        print(f"[{it}/{MAX_ITERS}] 학습손실 {running / EVAL_INTERVAL:.4f} | 판별 정확도: 검증 {va * 100:5.1f}% "
              f"(손실 {vl:.3f}) / 학습실제 {tr * 100:5.1f}% / 학습증강 {ta * 100:5.1f}% | {time.time() - t0:.0f}초{mark}")
        print(f"      선택 정확도 K=2 {s2['acc'] * 100:5.1f}% | K=4 {s4['acc'] * 100:5.1f}% "
              f"(정답 있는 문항 top1 {s4['top1_pos'] * 100:5.1f}%) | K=4 예측분포 {s4['pred_dist']}")
        running = 0.0

print(f"\n버린 샘플: 너무 김 {drop['long']} / 방해 줄 부족 {drop['short']}")

# ---------------- 최종 평가 (best 체크포인트) ----------------
ck = torch.load(OUT_CKPT, map_location=device)
model.load_state_dict(ck["model"])
va, vl = eval_pairs(val_pairs)
tr, _ = eval_pairs(diag_real)
ta, _ = eval_pairs(diag_aug)
s2, s4 = eval_select(val_sel[2]), eval_select(val_sel[4], keep_fails=6)
rule = {K: sum(rule_pick(i["q"], i["lines"]) == i["label"] for i in val_sel[K]) / N_EVAL for K in (2, 4)}
result = {
    "best_iter": ck["iter"],
    "pair_acc": {"val": va, "val_loss": vl, "train_real": tr, "train_aug": ta},
    "select_K2": {k: v for k, v in s2.items() if k != "fails"},
    "select_K4": {k: v for k, v in s4.items() if k != "fails"},
    "rule_baseline": {"K2": rule[2], "K4": rule[4]},
}
EVAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n===== 결과 =====")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n----- 틀린 예시 (K=4, 줄마다 P(일치)) -----")
for f in s4["fails"]:
    print(f"Q: {f['q']}  정답 {f['label']} / 예측 {f['pred']}")
    for i, (l, p) in enumerate(f["lines"]):
        print(f"    [{i + 1}] P={p:.3f}  {l}")
    print()