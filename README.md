# GameHikking (게임히끼잉)

리그 오브 레전드(LoL) 공략을 **RAG(검색 증강 생성)** 로 제공하는 챗봇.
일반 LLM이 최신 패치 수치를 지어내는 **환각** 문제를, 공식 데이터에서 근거 문서를 검색해 답하는 방식으로 줄이는 것이 목표다.

- 수업: 경성대학교 「인공지능」 2026-2학기 조별 과제
- 팀: 기미히끼잉 — 성현욱(팀장, 데이터 엔지니어링) / 이건우(AI·프롬프트) / 손영빈(백엔드·RAG) / 강동협(프론트엔드·발표)

> 최종 목표: 게임 중 *"상대 탑은 다리우스, 나는 나서스. 템트리 추천해줘"* 같은 질문에 아이템 순서·상성까지 근거 데이터 기반으로 답하는 코칭 봇. 현재는 패치·챔피언 정보 검색 단계.

---

## 두 트랙 구조

| 트랙 | 내용 |
|---|---|
| **실용 트랙 (RAG)** | 크롤링 → 정제(JSONL) → 임베딩 → Vector DB → 하이브리드 검색 → 응답 → Streamlit UI |
| **연구 트랙 (MiniGPT)** | 도메인 토크나이저 설계·학습 → PyTorch로 decoder-only Transformer 직접 구현(어텐션 포함) → 사전학습 → 근거 기반 응답 SFT 실험 |

---

## 디렉터리 구조

```
GameHikking/
├─ app.py                     # Streamlit 데모
├─ src/
│  ├─ data/                   # 데이터 수집·정제 파이프라인
│  │  ├─ crawler_all.py         # 공식 패치노트 전체 수집 (HTML 캐시)
│  │  ├─ build_patch_json.py    # 패치노트 HTML -> data/processed/patch_notes.jsonl
│  │  ├─ build_champions_json.py# Data Dragon -> data/processed/champions.jsonl
│  │  ├─ collect_lol_data.py    # 아이템·룬·주문·설정 (모드 태깅)
│  │  └─ make_corpus.py         # 토크나이저/사전학습 코퍼스
│  ├─ rag/                    # Vector DB 적재, 검색, 응답 파이프라인
│  ├─ gpt/                    # MiniGPT 본체
│  │  ├─ lol_tokenizer.py       # 도메인 토크나이저
│  │  ├─ mini_gpt.py            # 모델 (어텐션 직접 구현)
│  │  ├─ train_gpt.py           # 사전학습
│  │  ├─ build_sft_data.py      # SFT 데이터 생성
│  │  └─ experiments/           # SFT 실험 1~5c, 구조·커버리지 진단
│  └─ legacy/                 # 초기 버전 (기존 txt 산출물 재현용)
├─ data/
│  ├─ raw/                    # 원본 (크롤링 txt, 캐시)
│  ├─ processed/              # 정제된 JSONL
│  ├─ chunks/                 # 코퍼스, Data Dragon 추가 데이터
│  └─ sft/                    # SFT 학습/검증 데이터
├─ tokenizer_out/             # 학습된 토크나이저
├─ checkpoints/               # 모델 체크포인트(.pt, git 제외)·평가 결과(.json)
└─ Breifing/                  # 주간 발표 자료
```

---

## 환경

- macOS, Python 3.10 (Apple Silicon MPS 사용)
- 패키지: `requests beautifulsoup4 chromadb sentence-transformers tokenizers transformers torch streamlit`
- 유료 API 미사용 (임베딩: `distiluse-base-multilingual-cased-v1`, 로컬)

```bash
pip install requests beautifulsoup4 chromadb sentence-transformers tokenizers transformers torch streamlit
```

모든 스크립트는 **프로젝트 루트에서** 실행한다.

## 실행

```bash
# 1) 데이터 수집·정제
python src/data/crawler_all.py            # 패치노트 HTML 수집 (캐시가 있으면 재요청 안 함)
python src/data/build_patch_json.py       # -> data/processed/patch_notes.jsonl
python src/data/build_champions_json.py   # -> data/processed/champions.jsonl

# 2) 데모
streamlit run app.py

# 3) MiniGPT
python src/gpt/train_gpt.py                       # 사전학습
python src/gpt/experiments/diag_coverage.py       # 토큰 커버리지 진단
```

---

## 데이터

| 데이터 | 출처 | 규모 |
|---|---|---|
| 패치노트 | 공식 홈페이지 패치노트 (13.1 ~ 26.22, 91개 패치) | (패치, 챔피언) 1,439개 / 변경 4,199건 / 개발자 코멘트 약 29만 자 |
| 챔피언 정보 | Riot Data Dragon 16.22.1 | 173명 (능력치, 스킬, 쿨다운, 팁, 설정, 아이콘) |

`patch_notes.jsonl` 레코드 예시:
```json
{"id": "14-01_흐웨이", "patch": "14-01", "patch_order": 140100, "url": "...",
 "champion": "흐웨이", "summary": "기본 방어력이 증가합니다.", "context": "...",
 "changes": [{"slot": "기본 능력치", "skill": null, "field": "기본 방어력",
              "before": "18", "after": "21", "text": "기본 능력치: 기본 방어력: 18 ⇒ 21"}]}
```

---

## 연구 트랙: MiniGPT

- 구조: GPT-2 계열 decoder-only (pre-LayerNorm, GELU, weight tying, 학습형 위치 임베딩). Q/K/V 분할·scaled dot-product·causal mask를 직접 구현했으며 `nn.MultiheadAttention`이나 HuggingFace 모델 클래스는 쓰지 않았다.
- 설정: d_model 128, 레이어 4, 헤드 4, 파라미터 약 2.03M (문맥 128 기준)
- 토크나이저: Byte-level BPE (학습은 HuggingFace `tokenizers`), NFKC 정규화·숫자 한 자리 분리·도메인 용어 단일 토큰 등 설계는 직접, vocab 9,559
- 사전학습(처음부터): 학습 손실 3.86 / 검증 손실 4.52

### 근거 기반 응답 SFT 실험 (검증: 처음 보는 챔피언 20명)

| 실험 | 과제 형태 | 결과 |
|---|---|---|
| 1~4 | 근거 문장 생성(복사) | 정확 일치 0 |
| 커버리지 진단 | — | 검증 정답의 첫 토큰(챔피언명)이 100% 학습 데이터에 없음 → 출력 불가가 1차 원인 |
| 5 / 5b | 근거 4줄 중 번호 선택 | 학습·검증 모두 우연 수준 |
| 5c | 질문-근거 한 줄 일치 판별 | 판별 53% / 4지선다 25% (우연 수준), 규칙 기반 기준선 94% |

현재 서비스의 근거 선택은 규칙 기반으로 하고, MiniGPT는 사이드 트랙으로 개선을 이어간다 (재사전학습 → 리랭커 재시도).

---

## 진행 상황

- [x] 패치노트 크롤링·정제(JSONL), 챔피언 데이터 구조화
- [x] 도메인 토크나이저, MiniGPT 구현·사전학습, SFT 실험 및 원인 진단
- [ ] 검색 v2: JSONL 기반 Vector DB 재구축, 하이브리드 검색·동의어·최신 패치 우선, 테스트 쿼리 정량 평가
- [ ] 아이템 데이터 + 아이콘 카드 UI
- [ ] 두 트랙 통합 데모, 최종 보고서

---

GameHikking은 Riot Games가 보증하거나 후원하지 않는 비공식 학생 프로젝트이며, Riot Games 또는 리그 오브 레전드 제작·관리에 공식적으로 관여한 사람의 견해를 반영하지 않습니다. League of Legends 및 관련 데이터·이미지는 Riot Games, Inc.의 자산입니다.
