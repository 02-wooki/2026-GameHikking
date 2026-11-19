# src/gpt/train_sft5.py
"""실험 5: 근거 줄 '선택' SFT
근거 줄 K개에 [1]~[K] 번호를 붙이고, 모델은 질문에 맞는 줄 번호 하나만 출력한다 (맞는 줄이 없으면 0).
- 출력이 숫자 토큰 하나라서, 학습에 없던 토큰(홀드아웃 챔피언·스킬명)을 생성할 필요가 없다.
  (실험 4의 '정답 첫 토큰 출력 불가' 문제 회피. diag_coverage.py 결과: 검증 정답 첫 토큰 100% 미등장)
- 학습 샘플의 AUG_P 비율은 무작위 결합 증강: 학습 챔피언의 설명을 아무 챔피언 이름(홀드아웃 포함)과 섞는다.
  -> 암기 불가능. 홀드아웃 챔피언의 '이름 토큰'은 학습에 나오지만 '실제 근거 문장'은 학습에 없다.
- 커리큘럼: 근거 줄 수 2 -> 3 -> 4
- 검증: 홀드아웃 챔피언의 실제 문장만 사용 (K=2, K=4 두 세트)

실행 (프로젝트 루트에서, Mac 절전 방지):
  caffeinate -i /usr/local/bin/python3.10 src/gpt/train_sft5.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/gpt 모듈(mini_gpt 등) 찾기

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import build_sft_data as B
from lol_tokenizer import load as load_tokenizer, format_chat
from mini_gpt import MiniGPT

# ---------------- 설정 ----------------
ROOT = Path(__file__).resolve().parents[3]
OUT_CKPT = ROOT / "checkpoints" / "minigpt_sft5.pt"
EVAL_JSON = ROOT / "checkpoints" / "sft5_eval.json"

BLOCK_SIZE = 768
BATCH_SIZE, GRAD_ACCUM = 4, 4          # 유효 배치 16 (MPS 메모리 대응)
D_MODEL, N_LAYER, N_HEAD = 128, 4, 4
MAX_ITERS = 6000
EVAL_INTERVAL = 500
WARMUP = 200
LR, MIN_LR = 1e-3, 1e-4
AUG_P = 0.5                            # 학습 샘플 중 무작위 결합 증강 비율
NEG_P = 0.10                           # 맞는 줄이 없는 문항 비율 (정답 0)
TYPE_W = {"skill": 0.55, "stat": 0.10, "patch": 0.35}
N_EVAL = 300
SEED = 1337


def k_at(it):
    """커리큘럼: 반복 횟수에 따른 근거 줄 수"""
    return 2 if it <= 1500 else (3 if it <= 3000 else 4)


train_rng = random.Random(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 디바이스: {device}")

tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB = len(tok)
PAD_ID = tok.pad_token_id
END_ID = tok.convert_tokens_to_ids("<|end|>")

# 정답 숫자 토큰 확인 (숫자는 한 자리씩 분리되므로 0~4가 각각 토큰 1개여야 함)
ANS_IDS = {}
for i in range(5):
    ids = tok.encode(f"{i}<|end|>", add_special_tokens=False)
    assert len(ids) == 2 and ids[1] == END_ID, f"숫자 {i} 토큰화가 예상과 다름: {ids}"
    ANS_IDS[i] = ids[0]
print("정답 토큰:", {i: tok.convert_ids_to_tokens([t])[0] for i, t in ANS_IDS.items()})

# ---------------- 풀 구성 (train_sft4와 동일) ----------------
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
    """챔피언 5명 미만인 풀은 방해 줄을 못 만들므로 제외"""
    w = {}
    for i, t in enumerate(["skill", "stat", "patch"]):
        w[t] = TYPE_W[t] if len({x[0] for x in pools[i]}) >= 5 else 0.0
    return w


TRAIN_W, VAL_W = type_weights(train_pools), type_weights(val_pools)
print("유형 가중치 train:", TRAIN_W, "| val:", VAL_W)


# ---------------- 실제 데이터 문항 ----------------
def sname(slot, rng):
    return rng.choice(B.SLOT_NAMES.get(slot, [slot]))


def stat_line(c, s):
    return f"{c} 기본 능력치 - " + ", ".join(f"{k}: {s[k]}" for k in B.STAT_KEYS)


def fmt_patch(x):
    return f"{x[0]} {x[1]} 패치 - {x[3]}"


def real_skill(P, rng):
    sk = P[0]
    c, slot, line = rng.choice(sk)
    hard = [x for x in sk if x[0] == c and x[1] != slot]          # 같은 챔피언, 다른 스킬
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
    hard = [y for y in pt if y[0] == c and y[1] == pv and y[5] != key]   # 같은 패치, 다른 항목
    others = [y for y in rng.sample(pt, min(8, len(pt))) if y[0] != c]
    d = ([rng.choice(hard)] if hard and rng.random() < 0.5 else []) + others
    q = f"{c} {pv} 패치 {sname(slot, rng)} {field} 어떻게 바뀌었어?"
    return "patch", fmt_patch(x), [fmt_patch(y) for y in d], q


# ---------------- 무작위 결합 증강 (학습 챔피언의 설명만 사용) ----------------
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
    if rng.random() < 0.5:          # 같은 챔피언·패치, 다른 항목
        X = rng.choice([p for p in AUG_PT if qkey(p) != qkey(A)])
        d.append(patch_line(c, pv, X[0], X[1], rng.choice(AUG_PT)[2]))
    if rng.random() < 0.5:          # 같은 항목, 다른 챔피언
        d.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers), A[0], A[1], rng.choice(AUG_PT)[2]))
    while len(d) < 4:
        X = rng.choice(AUG_PT)
        d.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers), X[0], X[1], rng.choice(AUG_PT)[2]))
    return "patch", tgt, d, q


REAL = {"skill": real_skill, "stat": real_stat, "patch": real_patch}
AUG = {"skill": aug_skill, "stat": aug_stat, "patch": aug_patch}

# ---------------- 샘플 생성 ----------------
drop = {"long": 0, "short": 0}


def make_sample(P, rng, K, weights, aug=False):
    typ = rng.choices(list(weights), list(weights.values()))[0]
    kind, tgt, dists, q = (AUG if aug else REAL)[typ](P, rng)
    dists = [d for d in dists if d != tgt]
    neg = rng.random() < NEG_P
    need = K if neg else K - 1
    if len(dists) < need:
        drop["short"] += 1
        return None
    lines = dists[:need] + ([] if neg else [tgt])   # dists[0]은 어려운 방해 줄일 수 있음
    rng.shuffle(lines)
    label = 0 if neg else lines.index(tgt) + 1
    evid = "\n".join(f"[{i + 1}] {l}" for i, l in enumerate(lines))
    prompt = format_chat([{"role": "system", "content": evid},
                          {"role": "user", "content": q}], add_generation_prompt=True)
    p = tok.encode(prompt, add_special_tokens=False)
    t = [ANS_IDS[label], END_ID]
    if len(p) + len(t) > BLOCK_SIZE:
        drop["long"] += 1
        return None
    return {"kind": ("neg_" if neg else "") + kind, "K": K, "label": label, "question": q,
            "evidence": evid, "prompt_ids": p, "ids": p + t, "labels": [-100] * len(p) + t}


def sample_n(P, rng, n, K, weights, aug_p=0.0):
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 50:
            raise RuntimeError("샘플 생성 실패가 너무 많음 (풀 크기/BLOCK_SIZE 확인)")
        s = make_sample(P, rng, K, weights, aug=rng.random() < aug_p)
        if s:
            out.append(s)
    return out


val_sets = {K: sample_n(val_pools, random.Random(2024 + K), N_EVAL, K, VAL_W) for K in (2, 4)}
for K, items in val_sets.items():
    cnt = defaultdict(int)
    for v in items:
        cnt[v["kind"]] += 1
    print(f"검증 K={K} 문항 종류: {dict(sorted(cnt.items()))}")
ex = val_sets[4][0]
print("\n--- 검증 샘플 예시 (K=4) ---\n" + ex["evidence"] + f"\n[질문] {ex['question']}\n[정답] {ex['label']}\n")


def collate(items):
    L = max(len(it["ids"]) for it in items) - 1
    x = torch.full((len(items), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), L), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["ids"]) - 1
        x[i, :n] = torch.tensor(it["ids"][:-1])
        y[i, :n] = torch.tensor(it["labels"][1:])
    return x.to(device), y.to(device)


# ---------------- 모델 ----------------
model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                n_head=N_HEAD, block_size=BLOCK_SIZE).to(device)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


@torch.no_grad()
def evaluate(items, keep_fails=0):
    """정답 위치(프롬프트 마지막) 로짓으로 채점.
    free: 전체 vocab에서 argmax / con: 숫자 0~K 중에서만 argmax (제약 디코딩)"""
    model.eval()
    K = items[0]["K"]
    cand = torch.tensor([ANS_IDS[i] for i in range(K + 1)], device=device)
    loss_sum, free_ok, con_ok = 0.0, 0, 0
    by = defaultdict(lambda: [0, 0])
    fails = []
    for i in range(0, len(items), 8):
        batch = items[i:i + 8]
        L = max(len(b["prompt_ids"]) for b in batch)
        x = torch.full((len(batch), L), PAD_ID, dtype=torch.long)
        for j, b in enumerate(batch):
            x[j, :len(b["prompt_ids"])] = torch.tensor(b["prompt_ids"])
        x = x.to(device)
        last = torch.tensor([len(b["prompt_ids"]) - 1 for b in batch], device=device)
        lg = model(x)[torch.arange(len(batch), device=device), last]
        tgt = torch.tensor([ANS_IDS[b["label"]] for b in batch], device=device)
        loss_sum += F.cross_entropy(lg, tgt, reduction="sum").item()
        free = lg.argmax(-1)
        con_idx = lg[:, cand].argmax(-1)
        for j, b in enumerate(batch):
            f_ok = free[j].item() == tgt[j].item()
            c_ok = con_idx[j].item() == b["label"]
            free_ok += f_ok
            con_ok += c_ok
            by[b["kind"]][0] += c_ok
            by[b["kind"]][1] += 1
            if not c_ok and len(fails) < keep_fails:
                fails.append({"kind": b["kind"], "question": b["question"], "evidence": b["evidence"],
                              "label": b["label"], "pred": con_idx[j].item()})
    model.train()
    n = len(items)
    return {"loss": loss_sum / n, "free_acc": free_ok / n, "con_acc": con_ok / n,
            "by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by.items())}, "fails": fails}


# ---------------- 학습 ----------------
print("🚀 학습 시작 (매 반복 새 샘플, 커리큘럼 K=2→3→4)")
print(f"우연 수준: K=2 약 {1 / 2 * 100:.0f}~{1 / 3 * 100:.0f}% | K=4 약 {1 / 4 * 100:.0f}~{1 / 5 * 100:.0f}%")
model.train()
best, running, t0 = -1.0, 0.0, time.time()
for it in range(1, MAX_ITERS + 1):
    K = k_at(it)
    for g in opt.param_groups:
        g["lr"] = get_lr(it - 1)
    opt.zero_grad(set_to_none=True)
    for _ in range(GRAD_ACCUM):
        x, y = collate(sample_n(train_pools, train_rng, BATCH_SIZE, K, TRAIN_W, AUG_P))
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100) / GRAD_ACCUM
        loss.backward()
        running += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        e2, e4 = evaluate(val_sets[2]), evaluate(val_sets[4])
        mark = ""
        if e4["con_acc"] > best:
            best = e4["con_acc"]
            OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "iter": it, "val_k4_acc": best,
                        "config": dict(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                                       n_head=N_HEAD, block_size=BLOCK_SIZE)}, OUT_CKPT)
            mark = "  ← best 저장"
        print(f"[{it}/{MAX_ITERS}] K={K} 학습손실 {running / EVAL_INTERVAL:.4f} | "
              f"검증 K=2 정확도 {e2['con_acc'] * 100:5.1f}% (손실 {e2['loss']:.3f}) | "
              f"K=4 정확도 {e4['con_acc'] * 100:5.1f}% (손실 {e4['loss']:.3f}) | "
              f"{time.time() - t0:.0f}초{mark}")
        running = 0.0

print(f"\n버린 샘플: 너무 김 {drop['long']} / 방해 줄 부족 {drop['short']}")

# ---------------- 최종 평가 (best 체크포인트) ----------------
ck = torch.load(OUT_CKPT, map_location=device)
model.load_state_dict(ck["model"])
e2, e4 = evaluate(val_sets[2]), evaluate(val_sets[4], keep_fails=6)
result = {
    "best_iter": ck["iter"],
    "K2": {k: v for k, v in e2.items() if k != "fails"},
    "K4": {k: v for k, v in e4.items() if k != "fails"},
}
EVAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n===== 결과 (con_acc = 숫자 0~K 중 선택, free_acc = 전체 vocab 기준) =====")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n----- 틀린 예시 (K=4) -----")
for f in e4["fails"]:
    print(f"[{f['kind']}] Q: {f['question']}  정답 {f['label']} / 예측 {f['pred']}\n    "
          + f["evidence"].replace("\n", "\n    ")[:600] + "\n")