# src/gpt/diag_coverage.py
"""토큰 커버리지 측정
검증(홀드아웃 챔피언) 정답에 들어 있는 토큰 중, 학습 데이터에 한 번도 나오지 않은 토큰이 얼마나 되는지 센다.
- 학습에 안 나온 토큰은 임베딩이 학습되지 않고, weight tying 때문에 출력 확률도 계속 눌린다.
- 정답 첫 토큰(챔피언 이름)이 미등장이면 정확 일치는 구조적으로 불가능하다.

실행 (프로젝트 루트에서):
  /usr/local/bin/python3.10 src/gpt/diag_coverage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/gpt 모듈(mini_gpt 등) 찾기

import json
from collections import Counter
from pathlib import Path

from lol_tokenizer import load as load_tokenizer

ROOT = Path(__file__).resolve().parents[3]
tok = load_tokenizer(str(ROOT / "tokenizer_out"))
END_ID = tok.convert_tokens_to_ids("<|end|>")


def read(name):
    p = ROOT / "data" / "sft" / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def enc(s):
    return tok.encode(s, add_special_tokens=False)


train, val = read("sft_train.jsonl"), read("sft_val.jsonl")
print(f"학습 샘플 {len(train)}개 | 검증 샘플 {len(val)}개 | vocab {len(tok)}")

# 학습 데이터에서 본 토큰: 입력 어디든(any) / 정답으로(target)
seen_any, seen_tgt = Counter(), Counter()
for s in train:
    seen_any.update(enc(s["prompt"]))
    t = enc(s["target"])
    seen_any.update(t)
    seen_tgt.update(t)
print(f"학습 데이터에 등장한 토큰 종류: 입력·정답 전체 {len(seen_any)} / 정답으로만 {len(seen_tgt)}")

# 검증 정답(거절 제외) 분석
pos = [s for s in val if not s["kind"].startswith("neg_")]
tot = un_any = un_tgt = 0
smp_any = smp_tgt = first_any = first_tgt = 0
missing = Counter()
for s in pos:
    ids = [i for i in enc(s["target"]) if i != END_ID]
    if not ids:
        continue
    a = [i for i in ids if i not in seen_any]
    t = [i for i in ids if i not in seen_tgt]
    tot += len(ids)
    un_any += len(a)
    un_tgt += len(t)
    smp_any += bool(a)
    smp_tgt += bool(t)
    first_any += ids[0] not in seen_any
    first_tgt += ids[0] not in seen_tgt
    missing.update(a)

pct = lambda a, b: f"{a}/{b} ({a / b * 100:.1f}%)" if b else "0/0"
print("\n===== 검증 정답 토큰 커버리지 (거절 문항 제외) =====")
print("[토큰 단위]")
print("  학습 데이터 어디에도 없음      :", pct(un_any, tot))
print("  학습 정답으로는 한 번도 안 나옴 :", pct(un_tgt, tot))
print("[문항 단위]")
print("  미등장 토큰이 1개 이상인 문항(어디에도 없음):", pct(smp_any, len(pos)))
print("  미등장 토큰이 1개 이상인 문항(정답 기준)    :", pct(smp_tgt, len(pos)))
print("  정답 첫 토큰이 미등장(어디에도 없음)        :", pct(first_any, len(pos)))
print("  정답 첫 토큰이 미등장(정답 기준)            :", pct(first_tgt, len(pos)))

# 홀드아웃 챔피언 이름 자체
val_champs = sorted({s["champion"] for s in val})
print(f"\n[홀드아웃 챔피언 이름 {len(val_champs)}명]")
for c in val_champs:
    ids = enc(c)
    flag = "OK" if all(i in seen_any for i in ids) else "미등장"
    print(f"  {c:8s} 토큰 {len(ids)}개 {tok.convert_ids_to_tokens(ids)} -> {flag}")

print("\n[가장 많이 나온 미등장 토큰 상위 20]")
for i, n in missing.most_common(20):
    print(f"  {n:4d}회  {tok.decode([i])!r}")

print("\n해석 기준: 첫 토큰 미등장 비율이 높으면 정확 일치 0은 '외움'이 아니라 '출력 불가'가 1차 원인이다.")