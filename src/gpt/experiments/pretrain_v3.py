# 실험 6 사전학습: v2와 같은 코퍼스·설정, 모델 크기만 키움 (5e 과소적합 대응)
# src/gpt/experiments/pretrain_v3.py
"""MiniGPT 재사전학습 (사이드 트랙 ②)
코퍼스: patch_notes.jsonl(변경 줄 + 요약 + 개발자 코멘트) + champions.jsonl(스킬 설명, 팁, 설정)
- 검증 홀드아웃 챔피언 20명(sft_val.jsonl)의 레코드는 전부 제외 -> 이후 5d 평가가 '처음 보는 챔피언' 조건 유지
- 문서 단위로 95/5 학습/검증 분리, block 192 (실험 5c/5d와 같은 문맥 길이)
- 결과: checkpoints/minigpt_pretrain_v3.pt (검증 손실 최저 시점)

실행 (프로젝트 루트에서, 전원 연결):
  caffeinate -dims /usr/local/bin/python3.10 src/gpt/experiments/pretrain_v2.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from lol_tokenizer import load as load_tokenizer
from mini_gpt import MiniGPT

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "checkpoints" / "minigpt_pretrain_v3.pt"
BLOCK, BATCH = 192, 32
D_MODEL, N_LAYER, N_HEAD = 256, 6, 8   # v2: 128, 4, 4 (약 200만) -> 약 700만
MAX_ITERS, EVAL_EVERY, WARMUP = 8000, 500, 200
LR, MIN_LR = 1e-3, 1e-4
SEED = 1337

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = load_tokenizer(str(ROOT / "tokenizer_out"))
VOCAB = len(tok)
EOS = tok.convert_tokens_to_ids("<|end|>")


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


holdout = {json.loads(l)["champion"] for l in (ROOT / "data" / "sft" / "sft_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
docs = []
for r in load(PROC / "patch_notes.jsonl"):
    if r["champion"] in holdout:
        continue
    head = f"{r['champion']} {r['patch'].replace('-', '.')} 패치"
    body = [f"{head} - {c['text']}" for c in r["changes"]]
    note = " ".join(x for x in (r["summary"], r["context"]) if x)
    if note:
        body.append(f"{head} 변경 의도: {note}")
    docs.append("\n".join(body))
SLOT = {"패시브": "기본 지속 효과"}
for c in load(PROC / "champions.jsonl"):
    if c["name"] in holdout:
        continue
    lines = [f"{c['name']} ({c['title']}) 역할군: {', '.join(c['roles'])}"]
    lines += [f"{c['name']} {SLOT.get(s['slot'], s['slot'])} - {s['name']}: {s['description']}" for s in c["skills"]]
    lines += [f"{c['name']} 플레이 팁: {t}" for t in c["allytips"]] + [f"{c['name']} 상대 팁: {t}" for t in c["enemytips"]]
    if c.get("lore"):
        lines.append(f"{c['name']} 설정: {c['lore']}")
    docs.append("\n".join(lines))

random.shuffle(docs)
n_val = max(1, len(docs) // 20)
enc = lambda ds: torch.tensor([t for d in ds for t in tok.encode(d, add_special_tokens=False) + [EOS]], dtype=torch.long)
train, val = enc(docs[n_val:]), enc(docs[:n_val])
print(f"디바이스 {device} | 문서 {len(docs)}개 (홀드아웃 {len(holdout)}명 제외) | 학습 {len(train):,} 토큰 / 검증 {len(val):,} 토큰")
print(f"참고: 기존 사전학습 코퍼스는 45,598 토큰")


def batch(data):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def val_loss(n=20):
    model.eval()
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        ix = torch.randint(len(val) - BLOCK - 1, (BATCH,), generator=g)
        x = torch.stack([val[i:i + BLOCK] for i in ix]).to(device)
        y = torch.stack([val[i + 1:i + BLOCK + 1] for i in ix]).to(device)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    model.train()
    return tot / n


model = MiniGPT(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, block_size=BLOCK).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
lr_at = lambda it: LR * (it + 1) / WARMUP if it < WARMUP else \
    MIN_LR + 0.5 * (1 + math.cos(math.pi * (it - WARMUP) / (MAX_ITERS - WARMUP))) * (LR - MIN_LR)

best, run, t0 = float("inf"), 0.0, time.time()
for it in range(1, MAX_ITERS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it - 1)
    x, y = batch(train)
    loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    run += loss.item()
    if it % EVAL_EVERY == 0:
        vl = val_loss()
        mark = ""
        if vl < best:
            best = vl
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "iter": it, "val_loss": vl,
                        "config": dict(vocab_size=VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, block_size=BLOCK)}, OUT)
            mark = "  ← best 저장"
        print(f"[{it}/{MAX_ITERS}] 학습 {run / EVAL_EVERY:.4f} | 검증 {vl:.4f} | {time.time() - t0:.0f}초{mark}")
        run = 0.0
print(f"\n최고 검증 손실 {best:.4f} → {OUT}")