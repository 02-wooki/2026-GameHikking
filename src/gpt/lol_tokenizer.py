"""
LoL 꿀팁 AI용 토크나이저
- Byte-level BPE (한국어/영어/숫자/특수문자 혼합 데이터에서 UNK 없음)
- NFKC 정규화 (한글 자모 분리/전각 문자 정리)
- 챔피언/아이템/스킬 이름 등 도메인 용어를 단일 토큰으로 보존
- HuggingFace PreTrainedTokenizerFast로 감싸서 이후 모델 학습 코드에서 바로 사용 가능

사용법
------
# 학습
python lol_tokenizer.py train \
    --data ./data/chunks \
    --vocab-size 16000 \
    --terms ./data/lol_terms.txt \
    --out ./tokenizer_out

# 테스트
python lol_tokenizer.py test --tok ./tokenizer_out --text "아리 Q 맞추고 E로 이동기 빼면 돼"

데이터 형식 (--data 는 파일 또는 폴더)
- .jsonl : 한 줄당 {"text": "..."}  (키는 --text-key 로 변경 가능)
- .json  : [{"text": "..."}, ...] 또는 ["...", "..."]
- .txt   : 한 줄당 한 청크
"""
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# 스페셜 토큰 (이후 챗/파인튜닝 포맷에서 그대로 사용)
# ---------------------------------------------------------------------------
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
CHAT_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
# 크롤링 청크에 메타데이터를 붙일 때 쓸 수 있는 구분 토큰
META_TOKENS = ["<|champion|>", "<|role|>", "<|patch|>", "<|section|>"]

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS] + CHAT_TOKENS + META_TOKENS


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
def _iter_file(path: Path, text_key: str) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_key) if isinstance(obj, dict) else obj
                if text:
                    yield text
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            text = obj.get(text_key) if isinstance(obj, dict) else obj
            if text:
                yield text
    elif suffix == ".txt":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def iter_corpus(data_path: str, text_key: str = "text") -> Iterator[str]:
    p = Path(data_path)
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in {".jsonl", ".json", ".txt"}
    )
    if not files:
        raise FileNotFoundError(f"학습 데이터를 찾지 못했어요: {data_path}")
    for f in files:
        yield from _iter_file(f, text_key)


def load_terms(path: str | None) -> List[str]:
    """도메인 용어 파일(한 줄에 하나) 로드. 예: 챔피언/아이템/룬/스킬명"""
    if not path:
        return []
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = unicodedata.normalize("NFKC", line.strip())
            if t:
                terms.append(t)
    return sorted(set(terms))


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=UNK))
    # NFKC: 한글 NFD(자모 분리) -> 완성형으로 합쳐줌. 대소문자는 유지(Q/W/E/R, T1 등 의미 보존)
    tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
    # 숫자는 한 자리씩 분리(패치 수치/쿨다운/골드 계산에 유리), 나머지는 ByteLevel
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train(args) -> None:
    terms = load_terms(args.terms)
    tok = build_tokenizer()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 도메인 용어를 코퍼스에 한 번씩 추가로 흘려서 BPE가 기본적으로 잘 묶도록 유도
    def corpus() -> Iterator[str]:
        yield from iter_corpus(args.data, args.text_key)
        for t in terms:
            yield t

    tok.train_from_iterator(corpus(), trainer=trainer)

    # 도메인 용어는 일반 토큰(special=False)으로 추가 -> 항상 단일 토큰, decode 시 보존됨
    if terms:
        added = tok.add_tokens([AddedToken(t, special=False, normalized=True) for t in terms])
        print(f"도메인 용어 {added}개를 단일 토큰으로 추가했어요.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    # HF 호환 래퍼 저장 (config, special_tokens_map 등 생성)
    hf_tok = wrap_hf(tok)
    hf_tok.save_pretrained(str(out))
    print(f"저장 완료: {out}  (vocab size = {len(hf_tok)})")


def wrap_hf(tok: Tokenizer) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        unk_token=UNK,
        bos_token=BOS,
        eos_token=EOS,
        additional_special_tokens=CHAT_TOKENS + META_TOKENS,
        model_max_length=2048,
        clean_up_tokenization_spaces=False,
    )


def load(path: str) -> PreTrainedTokenizerFast:
    """모델 학습 코드에서: from lol_tokenizer import load; tok = load('./tokenizer_out')"""
    return PreTrainedTokenizerFast.from_pretrained(path)


# ---------------------------------------------------------------------------
# 채팅 포맷 헬퍼 (파인튜닝/추론 시 동일 포맷 유지용)
# ---------------------------------------------------------------------------
def format_chat(messages: List[dict], add_generation_prompt: bool = False) -> str:
    """messages = [{"role": "system|user|assistant", "content": "..."}]"""
    out = []
    for m in messages:
        out.append(f"<|{m['role']}|>{m['content']}<|end|>")
    if add_generation_prompt:
        out.append("<|assistant|>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
def test(args) -> None:
    tok = load(args.tok)
    texts = [args.text] if args.text else [
        "아리 Q 맞추고 E로 이동기 빼면 돼",
        "Patch 14.19: Ahri Q damage 40/65/90 -> 45/70/95",
        format_chat([
            {"role": "user", "content": "요네 라인전 팁 알려줘"},
        ], add_generation_prompt=True),
    ]
    for t in texts:
        enc = tok(t, add_special_tokens=False)
        ids = enc["input_ids"]
        print("-" * 60)
        print("원문   :", t)
        print("토큰   :", tok.convert_ids_to_tokens(ids))
        print("개수   :", len(ids), f"(글자 수 {len(t)} / 글자당 {len(ids)/max(len(t),1):.2f} 토큰)")
        decoded = tok.decode(ids)
        print("복원   :", decoded)
        print("일치   :", unicodedata.normalize("NFKC", t) == decoded)


def main():
    ap = argparse.ArgumentParser(description="LoL 꿀팁 AI 토크나이저")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True, help="청크 파일 또는 폴더 (.jsonl/.json/.txt)")
    tr.add_argument("--text-key", default="text")
    tr.add_argument("--terms", default=None, help="도메인 용어 파일 (챔피언/아이템/룬 등, 한 줄에 하나)")
    tr.add_argument("--vocab-size", type=int, default=16000)
    tr.add_argument("--min-frequency", type=int, default=2)
    tr.add_argument("--out", default="./tokenizer_out")
    tr.set_defaults(func=train)

    te = sub.add_parser("test")
    te.add_argument("--tok", default="./tokenizer_out")
    te.add_argument("--text", default=None)
    te.set_defaults(func=test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()