# src/gpt/experiments/train_frame11.py
"""실험 11: MiniGPT 질문 해석 모델 학습 (질문 + 이전 틀 -> 해석 틀)

설계: src/gpt/experiments/exp11_frame_spec.md / 공용 부품: src/gpt/frame_model.py
  pretrain_v4 몸통 + 필드별 분류 머리(범위·스킬·라인·역할군·난이도·시점·방향·개수·이어받음, 의도 11개 다중, 챔피언 역할)

학습 데이터: 선생님 LLM이 만든 JSONL (한 줄 = {"이전": {"질문","틀"}|null, "질문", "틀"}). 기본 경로 data/frame11/train.jsonl
평가: logs/eval/exp11_real_gold.jsonl (11주차 실제 질문 67개, 학습에 쓰지 않음) — frame_eval.score로 규칙 기준선과 같은 방식 채점
  ※ 실제 질문으로 체크포인트를 고르면 점수가 부풀므로 마지막 회차를 저장한다 (실험 10과 같은 원칙)

실행 (프로젝트 루트):
  caffeinate -dims /usr/local/bin/python3.10 -u src/gpt/experiments/train_frame11.py [--data 경로]
  코드 점검용(정답 세트에 일부러 과적합, 성능 아님): SMOKE=1 MAX_ITERS=200 python3.10 src/gpt/experiments/train_frame11.py
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "gpt"))
sys.path.insert(0, str(ROOT / "src" / "gpt" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "rag"))

import torch
import torch.nn.functional as F

import frame_eval as FE
from frame_model import HEADS, INTENTS, FrameModel, Masker, decode, to_example
from lol_tokenizer import load as load_tokenizer

INIT = ROOT / "checkpoints" / "minigpt_pretrain_v4.pt"
SMOKE = os.environ.get("SMOKE") == "1"
OUT_CKPT = ROOT / "checkpoints" / ("minigpt_frame11_smoke.pt" if SMOKE else "minigpt_frame11.pt")
EVAL_JSON = ROOT / "checkpoints" / "frame11_eval.json"
DATA = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else ROOT / "data" / "frame11" / "train.jsonl"
BLOCK_SIZE = 192
D_MODEL, N_LAYER, N_HEAD = 256, 6, 8
DROPOUT = 0.1
BATCH = 32
MAX_ITERS = int(os.environ.get("MAX_ITERS", 4000))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", 250))
WARMUP = 100
LR, MIN_LR = 3e-4, 3e-5
HEAD_LR_MULT = 10              # 새로 붙인 머리는 몸통보다 크게
SEED = 1337

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
torch.manual_seed(SEED)
tok = load_tokenizer(str(ROOT / "tokenizer_out"))
masker = Masker()
gold = FE.load_gold()


def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    exs, bad = [], 0
    for r in rows:
        ex = to_example(masker, tok, r, BLOCK_SIZE)
        if ex is None:
            bad += 1
        else:
            exs.append(ex)
    return exs, bad


if SMOKE:
    train, bad = load_rows(FE.GOLD)
    print(f"[SMOKE] 코드 점검: 정답 세트 {len(train)}개로 학습하고 같은 세트로 평가 (과적합 확인용, 성능 아님). 버린 줄 {bad}")
else:
    train, bad = load_rows(DATA)
    print(f"학습 {len(train)}개 ({DATA}), 버린 줄 {bad} | 평가: 실제 질문 {len(gold)}개")
assert train, "학습 데이터가 없습니다"

model = FrameModel(len(tok), D_MODEL, N_LAYER, N_HEAD, BLOCK_SIZE, DROPOUT).to(device)
ck0 = torch.load(INIT, map_location=device)
model.gpt.load_state_dict(ck0["model"])
head_params = [p for n, p in model.named_parameters() if not n.startswith("gpt.")]
opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": LR},
                         {"params": head_params, "lr": LR * HEAD_LR_MULT}], weight_decay=0.1)


def batchify(exs):
    L = max(len(e["ids"]) for e in exs)
    x = torch.full((len(exs), L), tok.pad_token_id, dtype=torch.long)
    for j, e in enumerate(exs):
        x[j, :len(e["ids"])] = torch.tensor(e["ids"])
    last = torch.tensor([len(e["ids"]) - 1 for e in exs])
    ph = [(j, p) for j, e in enumerate(exs) for _, p, _ in e["roles"]]
    ph_index = torch.tensor(ph, dtype=torch.long).reshape(-1, 2)
    return x.to(device), last.to(device), ph_index.to(device)


def loss_of(exs):
    x, last, ph_index = batchify(exs)
    out = model(x, last, ph_index)
    loss = sum(F.cross_entropy(out[k], torch.tensor([e["y"][k] for e in exs], device=device)) for k in HEADS)
    loss = loss + F.binary_cross_entropy_with_logits(
        out["의도"], torch.tensor([e["y"]["의도"] for e in exs], device=device)) * len(INTENTS) / 3
    loss = loss + F.cross_entropy(out["첫의도"], torch.tensor([e["y"]["첫의도"] for e in exs], device=device),
                                  ignore_index=-100) if any(e["y"]["첫의도"] >= 0 for e in exs) else loss
    roles = [r for e in exs for _, _, r in e["roles"]]
    if roles:
        loss = loss + 2 * F.cross_entropy(out["역할"], torch.tensor(roles, device=device))
    return loss


@torch.no_grad()
def predict(q, prev_frame):
    model.eval()
    from frame_model import encode               # 정답 없이 입력만 만든다
    table = {}
    prev_txt = masker.prev_text(prev_frame, table)
    qm, table = masker.mask(q, table)
    ids, pos = encode(tok, prev_txt, qm, BLOCK_SIZE)
    phs = [p for p in pos if p.startswith("[챔피언")]
    ex = {"ids": ids, "roles": [(p, pos[p], 0) for p in phs], "table": table}
    x, last, ph_index = batchify([ex])
    out = model(x, last, ph_index)
    one = {k: v[0] for k, v in out.items() if k != "역할"}
    f = decode(one, list(zip(phs, out["역할"])), ex)
    model.train()
    return f


def get_lr(it):
    if it < WARMUP:
        return (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return (MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)) / LR


rule_acc, _, _ = FE.score(gold, FE.rule_frame)
print(f"디바이스 {device} | 규칙 기준선 (실제 질문): 답 가능 해석 {rule_acc['답 가능 해석']:.1%}, 범위 {rule_acc['범위']:.1%}, "
      f"첫 의도 {rule_acc['첫 의도']:.1%}, 챔피언·역할 {rule_acc['챔피언·역할']:.1%}")
rng = random.Random(SEED)
running, t0 = 0.0, time.time()
model.train()
for it in range(1, MAX_ITERS + 1):
    m = get_lr(it - 1)
    opt.param_groups[0]["lr"] = LR * m
    opt.param_groups[1]["lr"] = LR * HEAD_LR_MULT * m
    loss = loss_of(rng.sample(train, min(BATCH, len(train))))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        acc, _, _ = FE.score(gold, predict)
        print(f"[{it}/{MAX_ITERS}] 손실 {running / EVAL_INTERVAL:.4f} | 실제 질문: 답 가능 해석 {acc['답 가능 해석']:.1%} "
              f"범위 {acc['범위']:.1%} 첫 의도 {acc['첫 의도']:.1%} 챔피언·역할 {acc['챔피언·역할']:.1%} "
              f"틀 전체 {acc['틀 전체']:.1%} | {time.time() - t0:.0f}초")
        running = 0.0

torch.save({"model": model.state_dict(), "iter": MAX_ITERS, "intents": INTENTS, "heads": HEADS,
            "config": dict(vocab_size=len(tok), d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, block_size=BLOCK_SIZE)},
           OUT_CKPT)
acc, per_scope, wrong = FE.score(gold, predict)
print(f"저장 {OUT_CKPT.name} (마지막 회차)")
for k, v in acc.items():
    print(f"  {k:10} {v:6.1%}   (규칙 {rule_acc[k]:6.1%})")
print("  범위별:", {f"{g}{'✓' if o else '✗'}": n for (g, o), n in sorted(per_scope.items())})
if not SMOKE:
    EVAL_JSON.write_text(json.dumps({"model": acc, "rule": rule_acc, "n_train": len(train), "iters": MAX_ITERS},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
