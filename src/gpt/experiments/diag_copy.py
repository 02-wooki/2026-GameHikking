# src/gpt/diag_copy.py
"""구조 진단: 완전 무작위 토큰으로 MiniGPT가 '복사'와 '조회'를 배울 수 있는지 확인.
외울 데이터가 없으므로, 성공하면 구조는 문제없고 / 실패하면 구조·학습 설정이 원인.

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/gpt/diag_copy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/gpt 모듈(mini_gpt 등) 찾기

import random
import time

import torch
import torch.nn.functional as F

from mini_gpt import MiniGPT

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("디바이스:", device)

VOCAB = 9559                 # 실제 모델과 동일한 vocab 크기 (임베딩/출력층 크기를 같게)
LO, HI = 100, 600            # 무작위 토큰은 500종만 사용 (우연 정답률 = 1/500)
SEP = 5
BLOCK, BATCH = 64, 32
STEPS, EVAL_EVERY = 8000, 500
LR = 1e-3


def gen_copy(rng, n, L=24):
    """s + SEP + s : SEP 뒤를 앞부분 그대로 복사"""
    xs, ms = [], []
    for _ in range(n):
        s = [rng.randrange(LO, HI) for _ in range(L)]
        xs.append(s + [SEP] + s)
        ms.append([0] * (L + 1) + [1] * L)
    return xs, ms


def gen_lookup(rng, n, K=4, KL=2, VL=6):
    """(키 값) K쌍 + SEP + 질문 키 + SEP + 해당 값 : 근거에서 이름으로 찾아 복사 (실제 과제와 같은 구조)"""
    xs, ms = [], []
    for _ in range(n):
        keys = set()
        while len(keys) < K:
            keys.add(tuple(rng.randrange(LO, HI) for _ in range(KL)))
        keys = list(keys)
        rng.shuffle(keys)
        vals = [[rng.randrange(LO, HI) for _ in range(VL)] for _ in range(K)]
        ctx = []
        for k, v in zip(keys, vals):
            ctx += list(k) + v
        q = rng.randrange(K)
        x = ctx + [SEP] + list(keys[q]) + [SEP] + vals[q]
        m = [0] * (len(ctx) + 1 + KL + 1) + [1] * VL
        xs.append(x)
        ms.append(m)
    return xs, ms


def to_tensors(xs, ms):
    x = torch.tensor(xs, dtype=torch.long)
    m = torch.tensor(ms, dtype=torch.bool)
    return x[:, :-1].to(device), x[:, 1:].to(device), m[:, 1:].to(device)


@torch.no_grad()
def evaluate(model, gen, rng, n=300):
    model.eval()
    x, y, m = to_tensors(*gen(rng, n))
    pred = model(x).argmax(-1)
    ok = (pred == y) | ~m                       # 마스크 밖은 무조건 통과
    tok_acc = ((pred == y) & m).sum().item() / m.sum().item()
    exact = ok.all(dim=1).float().mean().item()  # 정답 구간을 전부 맞힌 비율
    model.train()
    return tok_acc, exact


def run(name, gen):
    print(f"\n===== {name} =====")
    torch.manual_seed(0)
    rng, erng = random.Random(1), random.Random(999)
    model = MiniGPT(vocab_size=VOCAB, d_model=128, n_layer=4, n_head=4, block_size=BLOCK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time()
    last = (0, 0)
    for step in range(1, STEPS + 1):
        x, y, m = to_tensors(*gen(rng, BATCH))
        logits = model(x)
        loss = F.cross_entropy(logits[m], y[m])     # 정답 구간에만 손실
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            last = evaluate(model, gen, erng)
            print(f"[{step}/{STEPS}] 손실 {loss.item():.3f} | 토큰 정확도 {last[0]*100:5.1f}% | "
                  f"전부 일치 {last[1]*100:5.1f}% | {time.time()-t0:.0f}초")
    return last

c = (1.0, 1.0)   # 복사는 이전 실행에서 100% 확인됨 (생략)
run("조회 쉬운 버전 (키-값 2쌍)", lambda r, n: gen_lookup(r, n, K=2))
l = run("조회 (키-값 4쌍에서 찾기)", gen_lookup)

print("\n===== 요약 =====")
print(f"복사 전부 일치 {c[1]*100:.1f}% | 조회 전부 일치 {l[1]*100:.1f}%  (우연 수준 = 토큰 0.2%, 전부 일치 0%)")
if c[1] >= 0.9 and l[1] >= 0.9:
    print("→ 구조는 복사/조회를 배울 수 있다. 실제 데이터 실패 원인은 데이터(규모·다양성) 쪽일 가능성이 크다.")
elif c[1] >= 0.9:
    print("→ 단순 복사는 되지만 조회가 안 된다. 학습 반복/깊이 부족 가능성: STEPS를 늘려 재시도.")
else:
    print("→ 복사조차 못 배운다. 구조(위치 임베딩 등)나 학습 설정이 원인일 가능성: 구조 변경 실험으로 이동.")