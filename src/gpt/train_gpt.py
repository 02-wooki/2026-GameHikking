# src/gpt/train_gpt.py
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from lol_tokenizer import load as load_tokenizer
from mini_gpt import MiniGPT

# ---------------- 설정 ----------------
BATCH_SIZE = 16
BLOCK_SIZE = 128          # 기준선이라 기존 값 유지 (SFT 단계에서 키울 예정)
D_MODEL = 128
N_LAYER = 4
N_HEAD = 4
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP = 100
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_BATCHES = 50
VAL_RATIO = 0.1
SEG = 1024                # 검증 분리 단위(토큰). 세그먼트 단위로 train/val을 나눠 윈도우 겹침 방지
SEED = 1337
ROOT = Path(__file__).resolve().parents[2]   # src/gpt/train_gpt.py → 프로젝트 루트
CKPT_PATH = ROOT / "checkpoints" / "minigpt_pretrain.pt"
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"사용 디바이스: {device}")

# ---------------- 데이터 ----------------
tokenizer = load_tokenizer(str(ROOT / "tokenizer_out"))
vocab_size = len(tokenizer)
print(f"어휘 사전 크기: {vocab_size}")

text = (ROOT / "data" / "chunks" / "corpus.txt").read_text(encoding="utf-8")
ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"코퍼스 토큰 수: {len(ids):,}")

n_seg = len(ids) // SEG
perm = torch.randperm(n_seg)
n_val_seg = max(1, int(n_seg * VAL_RATIO))
val_segs, train_segs = perm[:n_val_seg], perm[n_val_seg:]
print(f"세그먼트: train {len(train_segs)} / val {len(val_segs)}  (1세그먼트={SEG}토큰)")


def get_batch(split):
    segs = train_segs if split == "train" else val_segs
    pick = segs[torch.randint(len(segs), (BATCH_SIZE,))]
    off = torch.randint(0, SEG - BLOCK_SIZE - 1, (BATCH_SIZE,))
    starts = (pick * SEG + off).tolist()
    x = torch.stack([ids[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([ids[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(device), y.to(device)


# ---------------- 모델 ----------------
model = MiniGPT(vocab_size=vocab_size, d_model=D_MODEL, n_layer=N_LAYER,
                n_head=N_HEAD, block_size=BLOCK_SIZE).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def get_lr(it):
    if it < WARMUP:
        return LEARNING_RATE * (it + 1) / WARMUP
    ratio = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * ratio)) * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            x, y = get_batch(split)
            logits = model(x)
            losses[k] = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_ckpt(it, val_loss):
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": dict(vocab_size=vocab_size, d_model=D_MODEL, n_layer=N_LAYER,
                       n_head=N_HEAD, block_size=BLOCK_SIZE),
        "iter": it,
        "val_loss": val_loss,
    }, CKPT_PATH)


# ---------------- 학습 ----------------
print("🚀 학습 시작...")
model.train()
best_val = float("inf")
running = 0.0

for it in range(1, MAX_ITERS + 1):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it - 1)

    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    running += loss.item()
    if it % 50 == 0:
        print(f"반복 [{it}/{MAX_ITERS}] | 학습 손실(50회 평균): {running / 50:.4f}")
        running = 0.0

    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        r = estimate_loss()
        mark = ""
        if r["val"] < best_val:
            best_val = r["val"]
            save_ckpt(it, best_val)
            mark = "  ← best 저장"
        print(f"  [평가 @{it}] train {r['train']:.4f} | val {r['val']:.4f}{mark}")

print(f"\n최고 검증 손실: {best_val:.4f}  (체크포인트: {CKPT_PATH})")


# ---------------- 생성 시연 ----------------
@torch.no_grad()
def generate_text(prompt, max_new_tokens=60, temperature=0.8, top_k=20):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -BLOCK_SIZE:])[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat((idx, nxt), dim=1)
    return tokenizer.decode(idx[0].tolist())


print("\n✨ 생성 시연:")
print(generate_text("나서스 Q"))