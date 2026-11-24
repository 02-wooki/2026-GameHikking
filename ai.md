# ai.md — GameHikking 프로젝트 요약 (AI 작업용)

> 기준일: **2026-11-24 (11주차 / 총 12주)**
> 출처: `README.md`, `Briefing/GameHikking_인수인계.md`(10월 중순 작성, 구버전), `Briefing/인수인계서_W11.md`(최신).
> 두 인수인계서 내용이 다르면 **W11이 우선**입니다. README의 MiniGPT 결과·진행 상황은 5c 시점에서 멈춰 있어 최신이 아닙니다.

---

## 1. 프로젝트 개요

- **이름:** GameHikking (게임히끼잉) / **팀:** 기미히끼잉
- **수업:** 경성대학교 「인공지능」 2026-2학기 조별 과제 (12주)
- **팀원:** 성현욱(팀장, 데이터 엔지니어링) / 이건우(AI·프롬프트) / 손영빈(백엔드·RAG) / 강동협(프론트엔드·발표)
- **한 줄 요약:** 리그 오브 레전드(LoL) 공략을 **RAG**로 제공하는 챗봇. 일반 LLM이 최신 패치 수치를 지어내는 환각 문제를 공식 데이터 검색으로 줄인다.
- **최종 유스케이스:** "상대 탑은 다리우스, 나는 나서스. 템트리 추천해줘" → 구매 순서·추천 아이템을 **아이콘 카드 UI**로, 근거 데이터 기반으로 답한다.

### 두 트랙 구조

| 트랙 | 내용 |
|---|---|
| **실용 트랙 (RAG)** | 크롤링 → 정제(JSONL) → 임베딩 → Chroma Vector DB → 하이브리드 검색 → 응답 → Streamlit UI, 그리고 Riot 매치 데이터 **집계** 기반 템트리 추천 |
| **연구 트랙 (MiniGPT)** | 도메인 토크나이저 설계·학습 → PyTorch로 decoder-only Transformer 직접 구현(어텐션 포함, `nn.MultiheadAttention`/HF 모델 클래스 미사용) → 사전학습 → "이 문장이 질문의 근거인가" 판별 SFT |

인공지능 수업이므로 **MiniGPT(직접 만든 AI)가 사실상 핵심 평가 포인트**입니다.

---

## 2. 협업 규칙 (반드시 지킬 것)

- **존댓말**로 답합니다. (구버전 인수인계서의 "반말" 메모는 틀린 내용입니다.)
- **투명성:** 사용자 모르게 설치·실행하지 않습니다. 명령은 무엇을 하는지 설명하고, 단순하고 되돌릴 수 있는 방법을 우선합니다.
- **유료 API 없음** (OpenAI 키 없음). 로컬/무료 방식을 우선합니다.
- **용어:** 템트리 추천은 학습이 아니라 **집계**입니다. "AI가 학습한 추천" 대신 "데이터 기반 템트리 추천"이라고 씁니다. 직접 학습시키는 AI는 MiniGPT뿐입니다.
- **사실과 다른 성과를 쓰지 않습니다.** 예: 토크나이저는 "직접 구현"이 아니라 "도메인 특화 토크나이저 설계 및 학습"(BPE 알고리즘 자체는 HF `tokenizers`가 수행).
- **보안:** `.env`(Riot API 키), `data/raw/riot/`, `seed_puuids.json`은 절대 커밋하지 않습니다. 키 내용을 채팅에 요청하지 않습니다. (과거 원본 매치 데이터가 공개 저장소에 올라간 사고가 있어, 저장소를 새 비공개 저장소로 다시 만들었습니다.)
- **발표:** 매주 진행 발표(즉석평가). 형식 고정 "지난주 계획 → 이번 주 한 일 → 다음 주 계획". 결과 중심, 혼자 읽어도 이해되는 자세한 표, 슬라이드 상단 내비게이션 pill. 교수님·학생에게는 비유(시험 공부하는 학생)로 설명하는 게 효과적이었습니다.

---

## 3. 환경

- 맥: `~/Documents/KSU/2026-2/GameHikking`, Python `/usr/local/bin/python3.10`, Apple Silicon MPS. 모든 스크립트는 **프로젝트 루트에서** 실행.
- 윈도우 PC: `py -3.10` (매치 수집 전용, 맥 디스크 부족 때문)
- 라즈베리파이 5 (배포 대상): 7.8GB RAM, Ubuntu 24.04, aarch64, Python 3.12, Cloudflare DNS + CloudPanel(nginx) 리버스 프록시, 도메인 `game.wooki.cloud`
- 임베딩: `distiluse-base-multilingual-cased-v1` (로컬, cosine). HF 경고 줄이기: `HF_HUB_OFFLINE=1`
- 학습 실행: `caffeinate -dims /usr/local/bin/python3.10 src/gpt/experiments/<파일>.py` (뚜껑 닫으면 잠들 수 있음)

---

## 4. 주요 파일

| 경로 | 역할 |
|---|---|
| `app.py` | Streamlit 앱. 템트리 질문이면 구매 순서를 아이템 아이콘 카드로 표시 |
| `src/rag/search.py` | 질문 해석(챔피언/패치/스킬 슬롯/의도, 별명·영문명·약어, 최신 변경 패치 규칙) + 검색. 근거 선택은 규칙 기반 `select_line()`, MiniGPT 연결 지점 `set_selector()`. `intent == "build"`면 `recommend()` 호출 |
| `src/rag/recommend.py` | 템트리 추천: 상대 피해 유형(Data Dragon info.attack/magic, 차이 2 이하면 혼합) + 집계 통계 + 방어 보너스 + 신발 + 구매 순서 |
| `src/rag/build_index.py` | Chroma 인덱스(`gamehikking_v2`) 생성 |
| `src/rag/eval_search.py` | 검색 평가 (dev 26, user 10, item 6 문항) |
| `src/data/crawler_all.py`, `build_patch_json.py`, `build_champions_json.py`, `build_items_json.py` | 패치노트·챔피언·아이템 수집/정제 → `data/processed/*.jsonl` |
| `src/data/collect_matches.py` | Riot 매치 수집기 (`--max --days --timeline --no-raw --seed-start`) |
| `src/data/aggregate_matches.py` | `participants.jsonl` → `data/processed/build_stats.json` |
| `src/gpt/` | `lol_tokenizer.py`, `mini_gpt.py`, `train_gpt.py`, `build_sft_data.py` |
| `src/gpt/experiments/` | 사전학습 v2/v3, SFT 1~6 실험, 진단 스크립트 |
| `checkpoints/` | `.pt`(gitignore), 평가 결과 `sft5e_eval.json`, `sft6_eval.json` |
| `Briefing/` | 주간 발표 자료, 인수인계서 |

참고: W11 인수인계서에 언급된 `probe_queues.py`, `probe_mayhem.py`는 현재 맥 저장소의 `src/data/`에 없습니다 (윈도우에만 있을 가능성).

---

## 5. 데이터 현황

| 데이터 | 출처 | 규모 |
|---|---|---|
| 패치노트 | 공식 홈페이지 (13.1 ~ 26.22, 91개 패치) | (패치, 챔피언) 1,439개 / 변경 4,199건 / 개발자 코멘트 약 29만 자 |
| 챔피언 | Data Dragon 16.22.1 | 173명 |
| 아이템 | Data Dragon 16.22.1 | 맵 ID 11 협곡, 12 칼바람, 30 아레나 (453 제외) |
| 매치 (Riot API) | League-V4 시드 11,000명(챌린저 300 + GM 700 + 마스터 10,000), Match-V5 큐 420, 최근 14일, 1인 20경기 | 마지막 보고(11/19): **총 22,066 매치, 시드 1044/11000** (윈도우에서 수집 중) |

- **집계 결과 (10주차 발표 기준):** 12,457경기, 선수 기록 124,570건, 패치 16.21~16.22, 챔피언×포지션 357, 맞대결 3,034개. 기준 MIN_GAMES 20 / MIN_MATCHUP 10 / TOP_ITEMS 8.
- 수집 재시작: `py -3.10 src\data\collect_matches.py --max 50000 --days 14 --seed-start <N>` (같은 키로 두 컴퓨터 동시 실행 금지, 개발 키는 24시간마다 만료)
- 동기화: 윈도우 `participants.jsonl` → 맥 복사 → `aggregate_matches.py` 재실행.

---

## 6. 결과 요약

### 검색 (top-1 근거 선택 정확도)
- dev 26문항 (v1 범위 23문항: 47.8% → **100%**, Wilson 하한 약 86%), user 10문항 (62.5% → **100%**), 아이템 **6/6**
- 블라인드 평가는 하지 않기로 결정 (발표에서도 언급 안 함).

### MiniGPT (근거 판별)
- 검증: 챔피언 20명을 학습·사전학습에서 완전히 제외한 홀드아웃 300문항. 규칙 기반 기준선 약 94%.
- **통과 기준(사전 고정, 변경 금지): 판별 ≥ 70% AND 4지선다 ≥ 50%**

| 실험 | 조건 | 파라미터 | 판별 | 4지선다 | 비고 |
|---|---|---|---|---|---|
| 1~4 | 근거 문장 생성(복사) | 2M | — | — | 정확 일치 0. 원인: 검증 정답 첫 토큰(챔피언명)이 학습 데이터에 없음 |
| 5c | 사전학습 없음 | 2M | 53.3% | 25.3% | 우연 수준 |
| 5d | pretrain_v2 (코퍼스 8.4배, 384,210 토큰) | 2M | 70.8% | 43.3% | 4지선다 미달 |
| **5e** | 5d + 15,000회 | 2M (d128/L4/H4) | **84.0%** | **61.3%** | **기준 첫 통과.** 학습 86%/검증 84% → 과소적합. 능력치(stat) 문항 0/19 |
| 6 | pretrain_v3 + sft6 | 7.2M (d256/L6/H8) | 67.2% | 38.7% | 기준 미달. 학습 77%/검증 67%, pretrain_v3 과적합 |
| **7** | 6 + 드롭아웃 (pretrain_v4 0.2, sft7 0.1) | 7.2M (d256/L6/H8) | **86.8%** | **70.3%** | **현재 최고 (15,000회 완주, best 12000회).** 2지선다 84.7%, 학습 91%/검증 87%. pretrain_v4 최저 검증 손실 2.4488 @4500 (v2 2.4910보다 낮음). 4지선다는 1000회 이후 64~70%에서 진동. 능력치 0/17 |
| (7 1차) | 위와 동일, 7500회에서 메모리 부족으로 중단 | 7.2M | 86.3% | 72.3% | best 5500회. 진동 폭 안의 높은 값으로 판단 → 발표·보고서 공식 수치는 완주한 7 사용. 파일 `sft7_run1_eval.json` |

(수치는 `checkpoints/sft5e_eval.json`, `sft6_eval.json`, `sft7_eval.json`과 일치 확인)

- 실험 7 틀린 예시 패턴: 질문의 **스킬 슬롯(Q/W/R)은 맞추지만 챔피언 이름을 무시**해, 다른 챔피언의 같은 슬롯 줄에 높은 확률을 줌. 능력치 문항은 여전히 거의 못 맞힘.

- **해석:** 작은 모델은 머리 용량 부족(과소적합), 큰 모델은 교재 부족(과적합). 데이터 없이 모델만 키우면 오히려 나빠진다.
- 실험 7로 확인: 큰 모델의 문제는 크기가 아니라 사전학습 과적합이었고, 드롭아웃으로 해결하면 큰 모델이 5e를 넘는다.
- (11/24 이전 권장안이었던 "5e 최종 확정"은 실험 7 결과로 대체될 후보. 최종 모델 선택은 사용자 확인 필요.) 서비스 근거 선택은 여전히 규칙 기반(94%)이 더 높음.

---

## 7. 현재 진행 중 / 다음 할 일 (11주차)

10주차 발표에서 약속한 것: 맞대결 통계 확충, 칼바람 모드 추천, MiniGPT 모델 키우기, 서버 배포·실시간 시연, 보고서 초안.

### A. 라즈베리파이 배포 — ✅ 완료 (2026-11-24)
**https://game.wooki.cloud** 에서 24시간 동작. 공인 IP·비밀번호는 발표 자료에 넣지 말 것.

**구성**: 브라우저 → Cloudflare(DNS+프록시, SSL Full strict) → 공유기 포트포워딩 → 파이5(Ubuntu 24.04, Wi-Fi) → nginx(CloudPanel 리버스 프록시, WebSocket 헤더) → `127.0.0.1:8510` Streamlit(`gamehikking.service`, 부팅 시 자동 시작·죽으면 5초 후 재시작). 앱은 로컬호스트에만 열려 있어 nginx 경유로만 접근 가능.

**한 일**
1. `build_stats.json` 재집계(702KB→841KB, 334게임) → rsync(`chroma_db/` 80MB, `data/processed/`, ddragon 챔피언)
2. 보안: 기존(ufw 기본 차단, SSH 키 인증만·root 금지, fail2ban) + 추가(fail2ban 단계적 차단 최대 1주·recidive, X11Forwarding 끔)
3. 정리: pink 서비스·pink용 GitHub Actions runner·FTP(proftpd) 중지 및 자동 시작 해제 → 21번 포트 닫힘, 파일 전송은 SSH(scp/rsync)만
4. 도메인: CloudPanel 리버스 프록시 + Cloudflare A 레코드 + Let's Encrypt
5. 외부 검증: HTTPS 200 / health ok / WebSocket 101 / http→https 301 / 실제 질문 답변 정상 / 재부팅 후 자동 복구 확인

**트러블슈팅 (발표 포인트)**
- **Wi-Fi 끊김**: 로그상 공유기(저가 ipTIME)가 접속 거부(status 16), 총 187회·9월과 11월 집중·최장 약 8시간 → 원인은 공유기. 절전 끄기 + 워치독(3분 주기, 조치 간 최소 10분, 30분 이상 끊기면 드라이버 재로딩). **교훈**: 첫 워치독이 1분마다 재연결을 시도해 오히려 40분 장애 유발 → 자동 복구는 쿨다운·단계적으로. 근본 해결은 유선 랜/공유기 교체.
- **시간 틀어짐**: RTC 배터리 없음(전원 끄면 1970년), 이 망에서 NTP는 time.bora.net만 응답 → chrony(bora 우선) + htpdate(HTTP 시간) 이중화, 재부팅 후 즉시 보정 확인. RTC 배터리 구매 권장.
- **도메인 502**: 앱·DNS는 정상, nginx 로그에서 `upstream 127.0.0.1:8000` 확인 → CloudPanel 기본 포트(8000)를 8510으로 수정. (`{{ }}`를 그대로 적어 문법 오류 났던 것도 수정)

**데모 체크리스트**
- 발표 10분 전 직접 접속해서 예열 (첫 로딩 약 20초, 임베딩 모델 로딩) → 예시 질문 한 번 눌러 확인
- 안 열리면 Wi-Fi 끊김 가능성 → 워치독이 수 분 내 복구, 급하면 파이 재부팅
- 백업: 노트북 로컬 `streamlit run app.py`

**발표 핵심 메시지 (제안)**: 저비용 자체 호스팅(파이 1대 + 무료 Cloudflare·Let's Encrypt) / 보안(내부 바인딩·키 인증·단계적 차단) / 운영 안정성(systemd·워치독·시간 동기화 이중화) / 실패에서 배운 점(과한 자동 복구) / 로그 기반 원인 특정

**파일·남은 것**
- `deploy/gamehikking.service`, `deploy/deploy_pi.sh`(재배포: `bash deploy/deploy_pi.sh`)
- ufw에 `8433:8443/tcp` 범위 허용이 있음 → 8443만 쓰므로 오타로 보임, 좁히기 권장
- 시연 후 공개 유지 여부 결정 필요

### B. 증강 칼바람 수집 (청중 질문 대응)
- 칼바람 참가자 기록에 `playerAugment1~6` 필드 존재 확인. 랭크 시드라 칼바람 표본 적음.
- `probe_mayhem.py` 윈도우에서 실행 대기 (큐 2400 = 증강 칼바람 후보, 미확정).
- 확인되면: `collect_matches.py`에 `--queue` 옵션 → 칼바람 집계 + 증강 선택률 → 모드 연동 추천. 증강 번호→이름 매핑 데이터가 따로 필요할 수 있음.

### C. 최종 보고서 초안 (12주차)
- 데이터·검색·추천·MiniGPT 결과 정리. MiniGPT는 **5c → 5d → 5e → 6** 흐름이 핵심 이야기.

### D. 기타
- 11주차 발표 덱: 10주차와 같은 스타일(남색 #0A1428 · 금색 #C8AA6E · 청록 #0AC8B9)로 새로 제작.
- (선택, 우선순위 낮음) 큰 모델 + 드롭아웃 또는 중간 크기(d192/L4) 재실험, 각 약 3시간.
- `.gitignore` 점검: `data/proceed` 오타 → `data/processed/`, `data/chunks/`, `data/sft/` 추가 권장.
- 미커밋 변경: `src/gpt/experiments/pretrain_v3.py`, `train_sft5e.py`, `train_sft6.py`(staged), `checkpoints/*_eval.json`, Briefing 신규 파일들.
- README의 MiniGPT 표·진행 상황 체크리스트가 5c 시점이라 갱신 필요 (5d/5e/6, 템트리 추천, 아이템 데이터 완료 반영).

---

## 8. 발표 자료 링크 (claude.ai 아티팩트)

| 이름 | 링크 |
|---|---|
| 10주차 덱 (9장) | https://claude.ai/artifact/EsZrieGBERUDia34EzzKHA |
| 10주차 대본 | https://claude.ai/artifact/9NAABgn6rBFJajjJoXEsf1 |
| 9주차 덱 | https://claude.ai/artifact/ViDmP5FSbmySex9C3wGdqX |
| 9주차 대본 | https://claude.ai/artifact/Kmg5ZzKPLERNtjLx76Fd5S |

10주차 덱을 수정할 때는 반드시 먼저 read 후 작업합니다.

---

## 9. 지난 흐름 (참고)

> 주차 구분은 두 인수인계서와 README에서 재구성한 것이라 5~9주차 경계는 대략적입니다. 정확한 내용은 `Briefing/`의 주차별 발표 자료를 확인하세요.

| 주차 | 내용 |
|---|---|
| 1~4 | 주제 선정, 패치노트 크롤링(구조 파싱·범위 제한으로 귀속 오류 해결), Data Dragon 챔피언 데이터, 다국어 임베딩 교체, 하이브리드 검색 |
| 5~8 | 91개 패치 전체 수집·JSONL 정제, 모드 태깅, 도메인 토크나이저(vocab 9,559), MiniGPT 구현·사전학습, SFT 1~5c 실패 및 원인 진단 |
| 9 | 검색 v2(별명·약어·최신 패치 규칙) 정량 평가 100%, Riot 매치 수집 시작 |
| 10 | 템트리 추천 완성(아이콘 카드 UI), 매치 집계, MiniGPT 5d/5e(기준 첫 통과), 실험 6 |
| 11 (현재) | 파이 배포 완료(game.wooki.cloud), 증강 칼바람 조사, 보고서 초안 |
| 12 | 최종 발표, 자팀·타팀 기술 보고서 |
