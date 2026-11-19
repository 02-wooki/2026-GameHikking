# src/gpt/train_sft2.py
"""무한 생성형 근거 복사 SFT.
근거 = 스킬/수치/패치 '한 줄' 2~3개 (정답 줄 + 다른 챔피언 방해 줄, 순서 무작위).
정답은 근거 줄을 그대로 복사한 문장. 정답 줄이 없으면 거절 문장.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/gpt 모듈(mini_gpt 등) 찾기

import json, math, random, re, unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import build_sft_data as B
from lol_tokenizer import load as load_tokenizer, format_chat
from mini_gpt import MiniGPT

# ---------------- 설정 ----------------
ROOT = Path(__file__).resolve().parents[3]
OUT_CKPT = ROOT / "checkpoints" / "minigpt_sft4.pt"
EVAL_JSON = ROOT / "checkpoints" / "sft4_eval.json"
NEG_P = 0.0

BLOCK_SIZE = 512
BATCH_SIZE, GRAD_ACCUM = 4, 4          # 유효 배치 16
D_MODEL, N_LAYER, N_HEAD = 128, 4, 4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
WARMUP = 150
LR, MIN_LR = 1e-3, 1e-4
NEG_P = 0.15
N_EVAL = 300
MAX_NEW = 250
SEED = 1337

train_rng = random.Random(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 디바이스: {device}")

tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB = len(tok)
PAD_ID = tok.pad_token_id
END_ID = tok.convert_tokens_to_ids("<|end|>")

# ---------------- 풀 구성 ----------------
base, patch = B.load_base(), B.load_patch()
sft_val = (ROOT / "data" / "sft" / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
val_champs = {json.loads(l)["champion"] for l in sft_val if l.strip()}
print(f"챔피언 {len({b['champ'] for b in base})}명 | 검증 홀드아웃 {len(val_champs)}명")


def sname(slot, rng):
    return rng.choice(B.SLOT_NAMES.get(slot, [slot]))

def build_pools(is_val):
    sk, st, pt = [], [], []
    for b in base:
        if (b["champ"] in val_champs) != is_val:
            continue
        for s in b["skills"]:
            sk.append((b["champ"], B.base_slot(s["slot"]), s["line"]))
        if all(k in b["stats"] for k in B.STAT_KEYS):
            st.append((b["champ"], b["stats"]))
    keys = defaultdict(int)
    cand = []
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
    pt = [c for c in cand if keys[c[5]] == 1]   # (챔피언, 패치, 슬롯, 줄, 필드) 유일한 것만
    return sk, st, pt


def stat_line(c, s):
    return f"{c} 기본 능력치 - " + ", ".join(f"{k}: {s[k]}" for k in B.STAT_KEYS)


def pick_skill(pools, rng):
    sk = pools[0]
    c, slot, line = rng.choice(sk)
    cand = [x for x in rng.sample(sk, 6) if not (x[0] == c and x[1] == slot)]
    if rng.random() < 0.4:
        same = [x for x in sk if x[0] == c and x[1] != slot]
        if same:
            cand.insert(0, rng.choice(same))
    dists = [f"{x[0]} {x[2]}" for x in cand][:3]
    q = rng.choice(B.SKILL_Q).format(c=c, s=sname(slot, rng))
    return "skill", f"{c} {line}", dists, q, f"{c} {line}"


def pick_stat(pools, rng):
    st = pools[1]
    c, s = rng.choice(st)
    key = rng.choice(B.STAT_KEYS)
    others = [x for x in rng.sample(st, 4) if x[0] != c][:3]
    q = rng.choice(B.STAT_Q).format(c=c, k=key)
    return "stat", stat_line(c, s), [stat_line(x[0], x[1]) for x in others], q, f"{c}의 {key}은(는) {s[key]}입니다."


def pick_patch(pools, rng):
    pt = pools[2]
    c, pv, slot, line, field, key = rng.choice(pt)
    cand = [x for x in rng.sample(pt, 6) if x[5] != key]
    if rng.random() < 0.5:
        hard = [x for x in pt if x[0] == c and x[1] == pv and x[5] != key]
        if hard:
            cand.insert(0, rng.choice(hard))
    fmt = lambda x: f"{x[0]} {x[1]} 패치 - {x[3]}"
    q = f"{c} {pv} 패치 {sname(slot, rng)} {field} 어떻게 바뀌었어?"
    return "patch", fmt((c, pv, slot, line, field, key)), [fmt(x) for x in cand][:3], q, fmt((c, pv, slot, line, field, key))

def make_sample(pools, rng, aug=False):
    neg = rng.random() < NEG_P
    typ = rng.choices(["skill", "stat", "patch"], [0.6, 0.0, 0.4])[0]
    if aug:
        kind, tgt, dists, q, ans = {
            "skill": lambda: aug_skill(rng, AUG_SK),
            "stat": lambda: aug_stat(rng),
            "patch": lambda: aug_patch(rng, AUG_PT),
        }[typ]()
    else:
        kind, tgt, dists, q, ans = {"skill": pick_skill, "stat": pick_stat,
                                    "patch": pick_patch}[typ](pools, rng)
    answer = B.REFUSE if neg else ans
    for nd in (0,):
        lines = dists[:nd] + ([] if neg else [tgt])
        if not lines:
            continue
        rng.shuffle(lines)
        evid = "\n".join(lines)
        prompt = format_chat([{"role": "system", "content": evid},
                              {"role": "user", "content": q}], add_generation_prompt=True)
        p = tok.encode(prompt, add_special_tokens=False)
        t = tok.encode(answer + "<|end|>", add_special_tokens=False)
        if len(p) + len(t) <= BLOCK_SIZE:
            return {"kind": ("neg_" if neg else "") + kind, "question": q, "answer": answer,
                    "evidence": evid, "prompt_ids": p, "ids": p + t,
                    "labels": [-100] * len(p) + t}
    return None

def sample_n(pools, rng, n, aug=False):
    out = []
    while len(out) < n:
        s = make_sample(pools, rng, aug)
        if s:
            out.append(s)
    return out

train_pools, val_pools = build_pools(False), build_pools(True)
all_names = sorted({b["champ"] for b in base} | {p["champ"] for p in patch})
patch_vers = sorted({p["patch"] for p in patch})


def parse_skills(sk_pool):          # (라벨, 스킬명, 설명)
    out = []
    for _, _, line in sk_pool:
        m = B.SKILL_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def parse_patch(pt_pool):           # (라벨, 필드, 값)
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


def aug_skill(rng, parts):
    A, D = rng.choice(parts), rng.choice(parts)
    c = rng.choice(all_names)
    tgt = skill_line(c, A[0], A[1], D[2])
    q = rng.choice(B.SKILL_Q).format(c=c, s=sname(B.base_slot(A[0]), rng))
    dists = []
    if rng.random() < 0.4:          # 같은 챔피언, 다른 슬롯
        X = rng.choice([p for p in parts if p[0] != A[0]])
        dists.append(skill_line(c, X[0], X[1], rng.choice(parts)[2]))
    if rng.random() < 0.5:          # 같은 슬롯, 다른 챔피언
        X = rng.choice(parts)
        dists.append(skill_line(other_name(rng, {c}), A[0], X[1], rng.choice(parts)[2]))
    while len(dists) < 3:
        X = rng.choice(parts)
        dists.append(skill_line(other_name(rng, {c}), X[0], X[1], rng.choice(parts)[2]))
    return "skill", tgt, dists[:3], q, tgt


def rand_stats(rng):
    return {"기본 체력": rng.randint(480, 720), "공격력": rng.randint(45, 75),
            "방어력": rng.randint(18, 48), "이동속도": rng.randint(325, 355)}


def aug_stat(rng):
    c = rng.choice(all_names)
    s = rand_stats(rng)
    key = rng.choice(B.STAT_KEYS)
    names = []
    while len(names) < 3:
        names.append(other_name(rng, {c, *names}))
    q = rng.choice(B.STAT_Q).format(c=c, k=key)
    return ("stat", stat_line(c, s), [stat_line(n, rand_stats(rng)) for n in names],
            q, f"{c}의 {key}은(는) {s[key]}입니다.")


def patch_line(name, pv, label, field, vals):
    return f"{name} {pv} 패치 - {label}: {field}: {vals}"


def qkey(p):                        # 질문이 가리키는 (슬롯, 필드)
    return (B.slot_of(p[0] + ": " + p[1]), p[1])


def aug_patch(rng, parts):
    A, V = rng.choice(parts), rng.choice(parts)
    c, pv = rng.choice(all_names), rng.choice(patch_vers)
    tgt = patch_line(c, pv, A[0], A[1], V[2])
    q = f"{c} {pv} 패치 {sname(qkey(A)[0], rng)} {A[1]} 어떻게 바뀌었어?"
    dists = []
    if rng.random() < 0.5:          # 같은 챔피언·패치, 다른 슬롯/필드
        X = rng.choice([p for p in parts if qkey(p) != qkey(A)])
        dists.append(patch_line(c, pv, X[0], X[1], rng.choice(parts)[2]))
    if rng.random() < 0.5:          # 같은 필드, 다른 챔피언
        dists.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers),
                                A[0], A[1], rng.choice(parts)[2]))
    while len(dists) < 3:
        X = rng.choice(parts)
        dists.append(patch_line(other_name(rng, {c}), rng.choice(patch_vers),
                                X[0], X[1], rng.choice(parts)[2]))
    return "patch", tgt, dists[:3], q, tgt
print(f"풀 크기 train(스킬/스탯/패치) {[len(x) for x in train_pools]} | val {[len(x) for x in val_pools]}")
val_items = sample_n(val_pools, random.Random(2024), N_EVAL)
print("검증 문항 종류:", dict(sorted(defaultdict(int, {k: sum(1 for v in val_items if v['kind'] == k)
                                                       for k in {v['kind'] for v in val_items}}).items())))
ex = val_items[0]
print("\n--- 샘플 예시 ---\n[근거]\n" + ex["evidence"] + f"\n[질문] {ex['question']}\n[정답] {ex['answer']}\n")


def collate(items):
    L = max(len(it["ids"]) for it in items) - 1
    x = torch.full((len(items), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), L), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["ids"]) - 1
        x[i, :n] = torch.tensor(it["ids"][:-1])
        y[i, :n] = torch.tensor(it["labels"][1:])
    return x.to(device), y.to(device)


# ---------------- 모델/학습 ----------------
model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                n_head=N_HEAD, block_size=BLOCK_SIZE).to(device)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def val_loss():
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(val_items), BATCH_SIZE):
        x, y = collate(val_items[i:i + BATCH_SIZE])
        n = (y != -100).sum().item()
        tot += loss_fn(model(x), y).item() * n
        cnt += n
    model.train()
    return tot / cnt


print("🚀 학습 시작 (매 반복 새 샘플 생성)...")
model.train()
best, running = float("inf"), 0.0
for it in range(1, MAX_ITERS + 1):
    for g in opt.param_groups:
        g["lr"] = get_lr(it - 1)
    opt.zero_grad(set_to_none=True)
    for _ in range(GRAD_ACCUM):
        x, y = collate(sample_n(train_pools, train_rng, BATCH_SIZE))
        loss = loss_fn(model(x), y) / GRAD_ACCUM
        loss.backward()
        running += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        vl = val_loss()
        mark = ""
        if vl < best:
            best = vl
            OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "iter": it, "val_loss": vl,
                        "config": dict(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                                       n_head=N_HEAD, block_size=BLOCK_SIZE)}, OUT_CKPT)
            mark = "  ← best 저장"
        print(f"[{it}/{MAX_ITERS}] 학습 답변손실 {running / EVAL_INTERVAL:.4f} | 검증 답변손실 {vl:.4f}{mark}")
        running = 0.0

print(f"\n최고 검증 답변손실: {best:.4f}")

# ---------------- 생성 평가 ----------------
model.load_state_dict(torch.load(OUT_CKPT, map_location=device)["model"])
model.eval()


@torch.no_grad()
def greedy(prompt_ids):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []
    for _ in range(MAX_NEW):
        if idx.size(1) >= BLOCK_SIZE:
            break
        nxt = model(idx)[:, -1, :].argmax(-1, keepdim=True)
        if nxt.item() == END_ID:
            break
        out.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    return out


NUM = re.compile(r"\d+(?:\.\d+)?")
norm = lambda s: unicodedata.normalize("NFKC", s).strip()
by_kind = defaultdict(lambda: [0, 0])
pos_n = pos_ref = neg_n = neg_ref = num_n = num_ok = 0
fails = []
print("\n생성 평가 중...")
for s in val_items:
    text = norm(tok.decode(greedy(s["prompt_ids"])))
    ok = text == norm(s["answer"])
    by_kind[s["kind"]][0] += ok
    by_kind[s["kind"]][1] += 1
    refused = "확인할 수 없" in text
    if s["kind"].startswith("neg_"):
        neg_n += 1
        neg_ref += refused
    else:
        pos_n += 1
        pos_ref += refused
        nums = set(NUM.findall(text))
        if nums:
            num_n += 1
            num_ok += nums <= set(NUM.findall(s["evidence"]))
    if not ok and len(fails) < 6:
        fails.append(s | {"gen": text})

tot_ok = sum(a for a, _ in by_kind.values())
result = {
    "val_answer_loss": best,
    "exact_match_total": f"{tot_ok}/{len(val_items)}",
    "exact_match_by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by_kind.items())},
    "refusal_correct_on_neg": f"{neg_ref}/{neg_n}",
    "false_refusal_on_pos": f"{pos_ref}/{pos_n}",
    "numbers_all_in_evidence": f"{num_ok}/{num_n}",
}
EVAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n===== 결과 =====")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n----- 틀린 예시 -----")
for s in fails:
    print(f"[{s['kind']}] Q: {s['question']}\n  근거:\n    " + s["evidence"].replace("\n", "\n    ")
          + f"\n  정답: {s['answer'][:150]}\n  생성: {s['gen'][:150]}\n")