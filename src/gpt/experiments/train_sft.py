# src/gpt/train_sft.py
"""근거 기반 SFT: <|system|>근거<|end|><|user|>질문<|end|><|assistant|>답변<|end|>
- 손실은 답변 토큰에만 건다 (프롬프트/근거 구간은 -100).
- 검증: 답변 손실 + 홀드아웃 챔피언 대상 greedy 생성 평가(정확 일치/수치 근거율/거절 정확도).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/gpt 모듈(mini_gpt 등) 찾기

import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from lol_tokenizer import load as load_tokenizer
from mini_gpt import MiniGPT

# ---------------- 설정 ----------------
ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = ROOT / "data" / "sft"
PRE_CKPT = ROOT / "checkpoints" / "minigpt_pretrain.pt"
OUT_CKPT = ROOT / "checkpoints" / "minigpt_sft.pt"
EVAL_JSON = ROOT / "checkpoints" / "sft_eval.json"

INIT = "scratch"          # "scratch" | "pretrain"
BLOCK_SIZE = 512
BATCH_SIZE = 4            # 마이크로 배치 (MPS 메모리 때문)
GRAD_ACCUM = 4            # 유효 배치 = 4 × 4 = 16
D_MODEL, N_LAYER, N_HEAD = 128, 4, 4
MAX_ITERS = 3000
EVAL_INTERVAL = 250
WARMUP = 100
LR = 1e-3 if INIT == "scratch" else 3e-4
MIN_LR = LR / 10
SEED = 1337
MAX_NEW = 400

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"사용 디바이스: {device} | 초기화: {INIT}")

# ---------------- 데이터 ----------------
tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB = len(tok)
PAD_ID = tok.pad_token_id
END_ID = tok.convert_tokens_to_ids("<|end|>")


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build(samples):
    items, dropped = [], 0
    for s in samples:
        p = tok.encode(s["prompt"], add_special_tokens=False)
        t = tok.encode(s["target"], add_special_tokens=False)
        assert t[-1] == END_ID, "target 끝이 <|end|> 토큰이 아님"
        if len(p) + len(t) > BLOCK_SIZE:
            dropped += 1
            continue
        items.append({"ids": p + t, "labels": [-100] * len(p) + t, "prompt_ids": p, "s": s})
    return items, dropped


train_items, d1 = build(load_jsonl(SFT_DIR / "sft_train.jsonl"))
val_items, d2 = build(load_jsonl(SFT_DIR / "sft_val.jsonl"))
print(f"train {len(train_items)} (제외 {d1}) | val {len(val_items)} (제외 {d2}) | vocab {VOCAB}")


def collate(items):
    L = max(len(it["ids"]) for it in items) - 1
    x = torch.full((len(items), L), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), L), -100, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["ids"]) - 1
        x[i, :n] = torch.tensor(it["ids"][:-1])
        y[i, :n] = torch.tensor(it["labels"][1:])
    return x.to(device), y.to(device)


def train_batches():
    while True:
        order = list(range(len(train_items)))
        random.shuffle(order)
        for i in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            yield collate([train_items[j] for j in order[i:i + BATCH_SIZE]])


# ---------------- 모델 ----------------
model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER,
                n_head=N_HEAD, block_size=BLOCK_SIZE).to(device)

if INIT == "pretrain":
    ck = torch.load(PRE_CKPT, map_location="cpu")
    assert ck["config"]["vocab_size"] == VOCAB, "vocab 크기 불일치"
    sd = ck["model"]
    old = sd["transformer.wpe.weight"]
    new = model.state_dict()["transformer.wpe.weight"].detach().cpu().clone()
    new[:old.size(0)] = old
    sd["transformer.wpe.weight"] = new
    model.load_state_dict(sd)
    print("사전학습 가중치 로드 (위치 임베딩 128→512 확장, 나머지 행은 새로 초기화)")

print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


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


# ---------------- 학습 ----------------
print("🚀 SFT 학습 시작...")
model.train()
gen = train_batches()
best, running = float("inf"), 0.0
for it in range(1, MAX_ITERS + 1):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it - 1)
    optimizer.zero_grad(set_to_none=True)
    it_loss = 0.0
    for _ in range(GRAD_ACCUM):
        x, y = next(gen)
        loss = loss_fn(model(x), y) / GRAD_ACCUM
        loss.backward()
        it_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    running += it_loss

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
        print(f"[{it}/{MAX_ITERS}] 학습 답변손실(평균) {running / EVAL_INTERVAL:.4f} | "
              f"검증 답변손실 {vl:.4f}{mark}")
        running = 0.0

print(f"\n최고 검증 답변손실: {best:.4f}")

# ---------------- 생성 평가 (best 체크포인트) ----------------
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
        t = nxt.item()
        if t == END_ID:
            break
        out.append(t)
        idx = torch.cat([idx, nxt], dim=1)
    return out


NUM = re.compile(r"\d+(?:\.\d+)?")


def norm(s):
    return unicodedata.normalize("NFKC", s).strip()


def evidence_of(prompt):
    return prompt.split("<|system|>", 1)[1].split("<|end|>", 1)[0]


print("\n생성 평가 중 (검증 전체 greedy, 몇 분 걸릴 수 있어)...")
by_kind = defaultdict(lambda: [0, 0])
pos_n = pos_refuse = neg_n = neg_refuse = 0
num_answers = num_grounded = 0
fails = []
for k, it in enumerate(val_items):
    s = it["s"]
    text = norm(tok.decode(greedy(it["prompt_ids"])))
    target = norm(s["target"].replace("<|end|>", ""))
    ok = text == target
    by_kind[s["kind"]][0] += ok
    by_kind[s["kind"]][1] += 1
    refused = "확인할 수 없" in text
    if s["kind"].startswith("neg_"):
        neg_n += 1
        neg_refuse += refused
    else:
        pos_n += 1
        pos_refuse += refused
        nums = set(NUM.findall(text))
        if nums:
            num_answers += 1
            num_grounded += nums <= set(NUM.findall(evidence_of(s["prompt"])))
    if not ok and len(fails) < 6:
        fails.append((s["kind"], s["question"], target, text))
    if (k + 1) % 50 == 0:
        print(f"  {k + 1}/{len(val_items)}")

result = {
    "val_answer_loss": best,
    "exact_match_by_kind": {k: f"{a}/{b}" for k, (a, b) in sorted(by_kind.items())},
    "exact_match_total": f"{sum(a for a, _ in by_kind.values())}/{len(val_items)}",
    "refusal_correct_on_neg": f"{neg_refuse}/{neg_n}",
    "false_refusal_on_pos": f"{pos_refuse}/{pos_n}",
    "numbers_all_in_evidence": f"{num_grounded}/{num_answers}",
}
EVAL_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n===== 결과 =====")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n----- 틀린 예시 -----")
for kind, q, tgt, gen_text in fails:
    print(f"[{kind}] Q: {q}\n  정답: {tgt[:200]}\n  생성: {gen_text[:200]}\n")