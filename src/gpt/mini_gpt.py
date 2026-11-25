# mini_gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from lol_tokenizer import load as load_tokenizer


# ---------------------------------------------------------------------------
# 1. 도메인 코퍼스 Dataset 클래스
# ---------------------------------------------------------------------------
class LoLDataset(Dataset):
    def __init__(self, corpus_path: str, tokenizer, block_size: int = 128):
        text = Path(corpus_path).read_text(encoding="utf-8")
        # 토크나이저로 전체 텍스트를 정수 ID 리스트로 변환
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        # 전체 토큰 길이에서 block_size 단위로 나눌 수 있는 윈도우 수
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        # 입력(x)과 정답(y - 한 칸 뒤로 시프트된 토큰)
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# 2. 미니 GPT 아키텍처 구성 요소
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Q, K, V 선형 변환 레이어 한 번에 정의
        self.c_attn = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model)
        # 드롭아웃 (실험 7부터 사용, 기본 0 = 이전 실험과 동일). 학습 파라미터가 없어 기존 체크포인트와 호환
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # Batch, Sequence length, Embedding dim

        # Q, K, V 계산 및 헤드 차원 분할
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 어텐션 스코어 계산 (Scaled Dot-Product Attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        # Causal Mask (미래 토큰을 보지 못하도록 상삼각행렬을 마스킹)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class Block(nn.Module):
    """트랜스포머 디코더 블록"""

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),   # 파라미터 없는 층이라 state_dict 키(mlp.0, mlp.2)는 그대로
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Residual Connection
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=4, n_head=4, block_size=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, d_model),
            wpe=nn.Embedding(block_size, d_model),
            h=nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 가중치 공유 (Weight Tying)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"입력 길이({T})가 최대 문맥 길이({self.block_size})를 초과했습니다."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits