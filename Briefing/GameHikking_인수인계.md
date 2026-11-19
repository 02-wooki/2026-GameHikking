# GameHikking 프로젝트 인수인계 문서

> **사용법**: 새 대화(다른 AI 포함)의 첫 메시지에 이 문서 전체를 붙여넣고, "이 프로젝트를 이어서 도와줘. 6장의 '다음 할 일' 1번부터 시작하자"라고 말하면 됩니다.
> 작성 시점: 2026-10-15 이후 어느 주차. 정확한 현재 주차는 사용자가 확인해서 알려줄 것 (구현 4~5주차 진도 수준).

---

## 0. 새 AI에게 (작업 방식 메모)

- 사용자는 **한국어 반말/캐주얼**로 대화하고, 응답도 한국어로 한다.
- 개발 환경: **macOS, PyCharm, Python 3.10 (`/usr/local/bin/python3.10`)**, 프로젝트 경로 `/Users/sunghyunwook/Documents/KSU/2026-2/GameHikking/`.
- 작업 흐름: AI가 코드를 주면 → 사용자가 PyCharm에 복붙해 실행 → **콘솔 출력/에러를 그대로 붙여넣음** → AI가 분석 후 다음 단계. 코드는 바로 실행 가능한 완결형으로 줄 것.
- 사용자는 **OpenAI API 키가 없다.** 유료 API 없이 무료/로컬로 가능한 방식을 우선할 것.
- 매주 수업에서 **진행상황 발표(즉석평가)**를 한다. 그래서 매 작업 후 "발표에 쓸 수 있는 결과물/스토리"까지 고려해 달라는 요구가 있었다 (PPT+대본을 매번 만들었음).
- 사용자가 시간에 쫓기는 경우가 많다(새벽 작업, 발표 직전). 길게 설명하기보다 **실행 가능한 코드 → 결과 확인 → 다음 단계** 순서가 선호된다.
- 이전 AI가 한 실수/주의: 사용자가 "트랜스포머 만들었다"고 했을 때 실제로는 **토크나이저 코드**였다. 모델 코드는 아직 없다. 사실과 다른 성과를 발표자료에 쓰지 않도록 주의할 것 (7장 "검증 필요 항목" 참고).

---

## 1. 프로젝트 개요

- **프로젝트명**: GameHikking (게임히끼잉) / **팀명**: 기미히끼잉
- **수업**: 대학 「인공지능」 조별 과제 (KSU, 2026-2학기, 총 12주)
- **팀원**: 성현욱(팀장, 데이터 엔지니어링) / 이건우(AI·프롬프트) / 손영빈(백엔드·RAG 파이프라인) / 강동협(프론트엔드·발표 총괄)
- **한 줄 요약**: 리그 오브 레전드(LoL) 공략을 RAG로 제공하는 챗봇. 일반 LLM의 환각(최신 패치/세부 수치를 지어냄) 문제를 출처 있는 문서 검색으로 해결.
- **게임**: 메인 = 리그 오브 레전드, 보조 = 스타크래프트 (RAG 파이프라인의 범용성 검증용 비교군. 현재 작업은 전부 LoL)

### 1.1 두 트랙 구조 (확정된 방향)

| 트랙 | 내용 | 목적 |
|---|---|---|
| **실용 트랙 (RAG)** | 크롤링 → 청킹 → 임베딩 → Vector DB → 검색 → LLM 답변 → UI | 실제로 쓸 수 있는 공략 서비스 |
| **연구 트랙 (미니 LLM)** | 도메인 토크나이저 → **PyTorch로 소형 트랜스포머(GPT 구조) 직접 구현** → 소규모 학습/생성 시연 | "원리를 이해하고 직접 구현했다"를 어필. 선배 말로는 LLM을 직접 만들면 A+ (평가기준에 명시된 건 아님) |

- 연구 트랙은 **"레벨 B"**: 처음부터 ChatGPT급을 만드는 게 아니라 nanoGPT 스타일의 작은 모델을 직접 짜서 LoL 텍스트로 학습시켜 보는 것.
- 소형 모델은 데이터가 적어서(수십만 자 수준) **문체 흉내 수준**이 한계. 실제 코칭 품질은 RAG + 사전학습 LLM이 담당하는 하이브리드 포지셔닝.
- 모델 부분에서 `transformers`의 `GPT2LMHeadModel` 같은 **완성 모델 클래스는 쓰지 않는다** (직접 구현 의미가 사라짐). `torch`만 사용.

### 1.2 최종 목표 유스케이스 (사용자가 확정)

1. 유저가 LoL을 플레이 중, 오버레이 공략(op.gg 데스크톱 등)을 켜지 않은 상태.
2. 도중에 "다음 아이템 뭐 사지?" 고민.
3. 서비스에 접속해 채팅으로 질문: *"상대 탑 라이너는 다리우스, 나는 나서스. 성능 괜찮은 템트리 추천해줘"*
4. **아이템 구매 순서 + 추천 아이템을 아이콘+명칭(UI)으로 제공.**
5. 핵심: 단순 너프/버프 목록을 뽑는 초급 봇이 아니라 **아이템 트리, 메타 챔피언, 상성을 고려한 플레이스타일까지 코칭하는 봇.** 단, 지금 바로 구현이 아니라 **기초를 먼저 다지고 최종 방향이 이것.**

AI가 합의한 설계 원칙:
- 추천 아이템/순서는 **LLM 기억이 아니라 데이터에서 계산해 가져온다** (안 그러면 환각 재발). LLM은 근거 데이터를 자연어로 설명하는 역할 → 답변이 "구조화 데이터(아이템 목록)+설명 텍스트"로 분리되어 아이콘 UI와 자연스럽게 연결.
- 필요한 추가 데이터: (a) 아이템 데이터+아이콘 — Data Dragon으로 쉬움, (b) **빌드 경로/상성/승률 — 어려움.** 공식 API로 집계 통계는 제공되지 않음. Riot Match API로 직접 집계(API 키+대량 수집 필요) 또는 op.gg 등 크롤링(이용약관 문제, 신중히). 아직 미착수.

---

## 2. 수업 평가 구조 (OT 내용)

1. 수업 절차: AI 분야 조사(개인, 수기, **완료**) / 팀 구성(구성원수 1~4점) / 주제 발표 / **매주 진행과정 발표(0·1·2점, 즉석평가)** / 최종발표
2. 중간 과정 평가: 10% + 보너스 (교수)
3. 최종 평가: **타팀 기술 보고서 20%(교수)** — 타팀 기술내용을 요약해 이해도 측정 + 자팀 기술 보고서 제출 / **타팀 평가** 학생 50% + 교수 10% / **팀원 자체평가** 학생 10%

주간 발표 포맷(고정): **지난주 계획 → 이번주 실제로 한 일 → 다음주 계획**.

---

## 3. 로드맵 (12주)

| 주차 | 내용 | 상태 |
|---|---|---|
| 1 | 주제 선정 | 완료 |
| 2~3 | 데이터 수집 설계, 크롤링 | 완료 |
| 4 | 크롤링 실제 수행 + 미니 트랜스포머 설계 착수 | 크롤링 완료, 토크나이저 코드 작성 |
| 5~6 | 청킹·임베딩·Vector DB (RAG) / 미니 트랜스포머 학습 코드 (연구) | RAG 완료, 연구 트랙 진행 중 |
| 7~8 | RAG 파이프라인 완성 / 미니 모델 소규모 학습·샘플 생성 | **다음 구간** |
| 9~10 | UI 구현, 두 트랙 통합 (RAG 메인 + 미니 LLM 데모 탭) | 예정 |
| 11 | 성능 비교·보완 | 예정 |
| 12 | 최종 발표 | 예정 |

사용자가 10/1에 작성한 계획: *(이번 주)* 단일 페이지 크롤링, 크롤링 데이터 청킹 → *(다음 주)* 여러 페이지 자동 크롤링, 챔피언 기본 데이터셋 세팅, DB 세팅, LLM 만들기 시작(토크나이저). **→ 다음 주 항목 전부 달성함.** 로드맵 대비 RAG 트랙은 약간 앞서 있음.

---

## 4. 현재까지 진행 상황 (완료된 것)

### 4.1 데이터 파이프라인 — 전부 실제 실행·검증 완료

| 단계 | 결과 | 산출 파일 |
|---|---|---|
| 패치노트 크롤링 | 5개 패치(26-16~26-20), **145건** 상향/하향 수치 | `crawled_data.txt` |
| 청킹 | 챔피언×패치 단위 **64개 청크** | `chunks.txt` |
| 챔피언 기본 데이터 | Data Dragon API로 **173명** | `champion_base_data.txt` |
| Vector DB | **237개 청크** (패치 64 + 기본정보 173) | `chroma_db/` |
| 검색 | 하이브리드 검색, 테스트 쿼리 **5/5 정답** | `vectorDB.py` |

### 4.2 겪은 문제와 해결 (발표/보고서의 소재)

1. **`<p>` 태그만 크롤링 → 상향/하향 수치 누락.** 개발자 도구로 구조 확인: 챔피언명=`<a>`, 스킬명=`<h4>`, 수치=`<li>`. 세 태그를 문서 순서대로 읽으며 현재 챔피언/스킬을 추적하도록 변경.
2. **후반부 전부 "볼리베어"로 귀속.** 패치노트 하단(VGU 비교·칼바람·버그수정·증강 등)의 구조가 달라 직전 챔피언에 계속 귀속됨. **해결: `h2#patch-champions` 아래, 다음 `h2` 전까지로 크롤링 범위 제한** (`find_all_next` + `h2`에서 break).
3. **정규식으로 "이름: " 접두어를 챔피언으로 추측하는 시도 → 실패** ("재사용 대기시간" 같은 필드 라벨이 챔피언명으로 오인됨). 추측 로직 제거, 범위 제한 방식 채택.
4. **청킹 파싱 에러** (`split("] ", 2)`가 `[patch_..][챔피언]` 사이에 공백이 없어 IndexError) → 정규식 `^\[patch_([^\]]+)\]\[([^\]]+)\]\s*(.+)$`로 교체.
5. **영어 전용 임베딩(`all-MiniLM-L6-v2`, Chroma 기본값)** → "나서스", "아펠리오스" 등 한국어 고유명사 검색 실패 → **다국어 모델로 교체.**
6. **임베딩 모델 다운로드 멈춤**(`paraphrase-multilingual-MiniLM-L12-v2`, 약 470MB) → 더 가벼운 `distiluse-base-multilingual-cased-v1`로 교체, 약 3분 내 완료.
7. **긴 질문에서 챔피언명 신호 희석** ("나서스 26-19 패치에서 뭐 바뀌었어?"에서 엉뚱한 챔피언 검색) → **질문에서 챔피언명·패치번호를 먼저 감지해 메타데이터 `where` 필터 적용 후 벡터 검색(하이브리드).**
8. **(진행 중)** 아이템 데이터에 칼바람/아레나 등 타 모드 전용 아이템이 섞여 있음 → **모드 태깅 방식으로 설계 중** (5장 참고).

### 4.3 알려진 한계

- **역할군 조건 검색 약함**: "탱커 챔피언 추천해줘"처럼 챔피언명이 없는 속성 질문은 순수 벡터 검색이라 Fighter/Assassin이 섞임. 개선안: `역할군: Tank` 텍스트/메타데이터 직접 필터.
- **임베딩 모델 입력 길이 제한(추정, 확인 필요)**: `distiluse-base-multilingual-cased-v1`는 최대 약 128토큰으로 알려져 있어, 긴 청크(예: 카시오페아 26-18)는 뒷부분이 잘려 임베딩될 수 있음. 아이템/룬 데이터를 넣을 때 청크를 짧게 유지할 것.
- 패치 데이터는 **최근 5개 패치**뿐. 커뮤니티 공략/빌드/상성 데이터 없음.
- 챔피언 기본정보 텍스트에 `<br>` 등 HTML이 남아 있어 정제 필요 (DB 적재 시 `<br>`만 제거했음).

---

## 5. 지금 진행 중이던 작업: 데이터 확장 + 모드 태깅

### 5.1 합의된 설계

- 목적: 토크나이저/소형 LLM 학습 코퍼스 확대 + 최종 유스케이스용 아이템/룬/주문/챔피언 팁 수집.
- **모든 모드(협곡·칼바람·증강 칼바람·TFT 등)를 버리지 않고 `mode` 메타데이터로 구분 저장.** RAG 답변 시 모드를 구분해 답하거나 UI에서 선택하게 함.
- 검색 필터: `where={"mode": {"$in": [선택모드, "공통"]}}` → 선택 모드 + 공통 데이터만 검색.
- 기존 데이터에도 `mode` 부여 필요: 패치노트 챔피언 섹션은 `소환사의 협곡`으로 간주(가정), 챔피언 기본정보/설정/룬은 `공통`.

### 5.2 모드 구분의 함정 (반드시 인지)

1. **칼바람과 증강 칼바람은 같은 맵 ID(`12`)** 를 쓴다. 아이템은 "칼바람"으로 묶어 저장하고, **증강 자체 데이터는 `item.json`에 없어 별도 수집 필요.**
2. **TFT는 `item.json`에 없음.** `tft-item.json`, `tft-champion.json`, `tft-trait.json` 존재/구조를 먼저 확인해야 함(아래 확인 스니펫). 구조 미확인 상태에서 추측 코드 작성 금지.
3. **맵 ID↔모드 대응(`11`=협곡, `12`=칼바람, `30`=아레나)은 AI의 기억값. 소환사 주문의 `modes` 값(`CLASSIC`/`ARAM`/`CHERRY`)도 미검증.** 스크립트 출력(맵 ID별 개수)으로 검증하고 대응표를 확정해야 함. 대응표에 없는 키는 저장하지 않고 개수만 출력하도록 해둠.
4. **Chroma 메타데이터는 `None` 값을 허용하지 않는다.** 아래 스크립트의 `icon=None` 레코드를 DB에 넣을 때는 None 키를 제거할 것.

### 5.3 수집 스크립트 v2 (`collect_lol_data.py`) — **작성만 됨, 실행 결과 미확인**

```python
# collect_lol_data.py  (v2: 모드 태깅)
import json
import re
import requests
from collections import Counter
from pathlib import Path

BASE = "https://ddragon.leagueoflegends.com"
OUT = Path("data/chunks")
OUT.mkdir(parents=True, exist_ok=True)

# 맵 ID -> 모드. 실행 후 출력으로 반드시 검증할 것
MAP_MODES = {"11": "소환사의 협곡", "12": "칼바람", "30": "아레나"}
# 소환사 주문의 modes 값 -> 모드
SPELL_MODES = {"CLASSIC": "소환사의 협곡", "ARAM": "칼바람", "CHERRY": "아레나"}
COMMON = "공통"


def clean(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", t).strip()


ver = requests.get(f"{BASE}/api/versions.json", timeout=10).json()[0]
print("버전:", ver)

records, terms = [], set(MAP_MODES.values()) | {"증강 칼바람", "전략적 팀 전투"}


def add(text, mode, typ, name, icon=None):
    records.append({"text": f"[{mode}] {text}", "mode": mode,
                    "type": typ, "name": name, "icon": icon})


# 1. 아이템: 활성화된 맵마다 모드별 레코드 생성
items = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/item.json", timeout=10).json()["data"]
id2name = {k: v["name"] for k, v in items.items()}
map_counter = Counter()

for iid, it in items.items():
    if not it.get("gold", {}).get("purchasable", False):
        continue
    name = it["name"]
    terms.add(name)
    recipe = [id2name[i] for i in it.get("from", []) if i in id2name]
    icon = f"{BASE}/cdn/{ver}/img/item/{it['image']['full']}"
    body = f"아이템 {name}: {clean(it.get('description', ''))} 총 가격 {it['gold']['total']}골드."
    if recipe:
        body += f" 조합식: {' + '.join(recipe)}."
    for map_id, enabled in it.get("maps", {}).items():
        if not enabled:
            continue
        map_counter[map_id] += 1
        mode = MAP_MODES.get(map_id)
        if mode:
            add(body, mode, "item", name, icon)

print("\n[아이템] 맵 ID별 개수 (대응표에 없는 키는 저장 안 됨):")
for k, v in sorted(map_counter.items()):
    print(f"  맵 {k}: {v}개  -> {MAP_MODES.get(k, '미분류(저장 안 함)')}")

# 2. 소환사 주문
spells = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/summoner.json", timeout=10).json()["data"]
for sp in spells.values():
    terms.add(sp["name"])
    icon = f"{BASE}/cdn/{ver}/img/spell/{sp['image']['full']}"
    for m in sp.get("modes", []):
        if m in SPELL_MODES:
            add(f"소환사 주문 {sp['name']}: {clean(sp['description'])}",
                SPELL_MODES[m], "spell", sp["name"], icon)

# 3. 룬 (모드 구분 정보 없음 -> 공통)
runes = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/runesReforged.json", timeout=10).json()
for tree in runes:
    for slot in tree["slots"]:
        for r in slot["runes"]:
            terms.add(r["name"])
            add(f"룬 {r['name']} ({tree['name']}): {clean(r['longDesc'])}",
                COMMON, "rune", r["name"])

# 4. 챔피언 설정과 팁 (공통)
champs = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/champion.json", timeout=10).json()["data"]
for key in champs:
    d = requests.get(f"{BASE}/cdn/{ver}/data/ko_KR/champion/{key}.json",
                     timeout=10).json()["data"][key]
    name = d["name"]
    terms.add(name)
    add(f"{name} 설정: {clean(d['lore'])}", COMMON, "lore", name)
    for tip in d.get("allytips", []):
        add(f"{name} 플레이 팁: {clean(tip)}", COMMON, "ally_tip", name)
    for tip in d.get("enemytips", []):
        add(f"{name} 상대 팁: {clean(tip)}", COMMON, "enemy_tip", name)

Path("data/chunks/ddragon_extra.txt").unlink(missing_ok=True)   # 구버전 중복 방지
with open(OUT / "ddragon_extra.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

term_path = Path("data/lol_terms.txt")
old = set(term_path.read_text(encoding="utf-8").splitlines()) if term_path.exists() else set()
term_path.write_text("\n".join(sorted(old | terms)), encoding="utf-8")

print("\n[모드 x 종류별 레코드 수]")
for (mode, typ), n in sorted(Counter((r["mode"], r["type"]) for r in records).items()):
    print(f"  {mode:10s} {typ:10s} {n}")
print(f"\n총 {len(records)}건 / {sum(len(r['text']) for r in records):,}자, 용어 {len(old | terms)}개")
```

### 5.4 TFT 구조 확인 스니펫 (미실행)

```python
import requests
ver = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
for fn in ["tft-item.json", "tft-champion.json", "tft-trait.json"]:
    r = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/ko_KR/{fn}")
    print(fn, r.status_code)
    if r.status_code == 200:
        data = r.json()["data"]
        first = next(iter(data.values()))
        print("  개수:", len(data), "| 첫 항목 키:", list(first.keys()))
```

### 5.5 토크나이저 학습용 코퍼스 준비 (`make_corpus.py`) — **실행 여부 불명 (사용자에게 확인)**

`champion_base_data.txt`와 `chunks.txt`를 줄 단위 코퍼스(`data/chunks/corpus.txt`)로 합치고, 챔피언명·스킬명을 `data/lol_terms.txt`로 추출한다.

```python
# make_corpus.py
import re
from pathlib import Path

Path("data/chunks").mkdir(parents=True, exist_ok=True)
corpus_lines, terms = [], set()
skill_re = re.compile(r"^(?:[QWER]|기본 지속 효과) - ([^:]+):")

with open("champion_base_data.txt", encoding="utf-8") as f:
    raw = f.read()
for block in raw.strip().split("=== ")[1:]:
    champion, body = block.split(" ===\n", 1)
    terms.add(champion.strip())
    body = re.sub(r"<br\s*/?>", " ", body)
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        corpus_lines.append(line)
        m = skill_re.match(line)
        if m:
            terms.add(m.group(1).strip())

with open("chunks.txt", encoding="utf-8") as f:
    raw = f.read()
for block in raw.strip().split("=== ")[1:]:
    header, body = block.split(" ===\n", 1)
    terms.add(header.split("|")[0].strip())
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        corpus_lines.append(line)
        m = skill_re.match(line)
        if m:
            terms.add(m.group(1).strip())

Path("data/chunks/corpus.txt").write_text("\n".join(corpus_lines), encoding="utf-8")
Path("data/lol_terms.txt").write_text("\n".join(sorted(terms)), encoding="utf-8")
print(f"코퍼스 {len(corpus_lines)}줄, 도메인 용어 {len(terms)}개 저장 완료")
```

**실행 순서**: `make_corpus.py` → `collect_lol_data.py`(용어 사전을 덮어쓰지 않고 합침) → 토크나이저 학습.

---

## 6. 다음 할 일 (우선순위)

1. **데이터 수집 실행·검증**: `make_corpus.py` → `collect_lol_data.py` 실행, "맵 ID별 개수"와 "모드×종류별 레코드 수" 출력 확인 → 맵 ID/모드 대응표 확정 (특히 `21`, `22` 등 미분류 키 처리). TFT 구조 확인 스니펫 실행.
2. **토크나이저 실제 학습·검증** (코드는 있음, 학습 결과 미확인):
   ```bash
   pip install tokenizers transformers requests
   python lol_tokenizer.py train --data ./data/chunks --terms ./data/lol_terms.txt --vocab-size 8000 --out ./tokenizer_out
   python lol_tokenizer.py test --tok ./tokenizer_out --text "나서스 Q 흡수의 일격 추가 피해량 40/60/80/100/120"
   ```
   확인: "나서스"/"흡수의 일격"이 단일 토큰인지, 숫자가 한 자리씩 분리되는지, 마지막에 `일치: True`. 발표용으로 GPT-2 토크나이저(`AutoTokenizer.from_pretrained("gpt2")`)와 **같은 문장의 토큰 수 비교**를 해 도메인 토크나이저의 효과를 수치로 제시. `--vocab-size`는 코퍼스가 작으므로 8000 권장(기본 16000은 과대).
3. **Vector DB 재구축 (모드·아이템 포함)**: 모든 레코드에 `mode`, `type`, `name`, (`icon`), (`champion`, `patch`) 메타데이터 부여. **None 값 제거.** 짧은 청크 유지. 검색은 `where={"mode": {"$in": [모드, "공통"]}}` + 기존 챔피언/패치 감지 로직 결합.
4. **RAG 응답 생성 프로토타입**: 검색 결과 → LLM → 자연어 답변. 사용자는 OpenAI 키 없음. 후보: 로컬 LLM(Ollama), 무료 티어 API 등 (사용자와 상의). 답변은 반드시 검색된 근거에만 기반하도록 프롬프트 구성, 출처(챔피언/패치/모드) 표기.
5. **미니 트랜스포머 구현 (연구 트랙 본체)**: PyTorch로 decoder-only GPT 구조(토큰+위치 임베딩, causal self-attention, MLP, LayerNorm, 잔차)를 직접 구현. 코퍼스가 수십만 자 수준이므로 **아주 작은 설정**(예: 레이어 4, 헤드 4, d_model 128~256, 문맥 길이 128~256)에서 시작 — 이 수치는 제안일 뿐, 코퍼스 크기(토큰 수)를 확인하고 조정할 것. 학습 루프(AdamW, 교차엔트로피), 샘플 생성(temperature/top-k) 포함. `transformers`의 완성 모델 클래스 사용 금지.
6. **역할군 조건 검색 개선** ("탱커 추천해줘" 등) — 우선순위 낮음.
7. **아이템 추천 데이터(빌드/상성/승률)**: Riot Match API 직접 집계 vs 외부 사이트 크롤링(약관 주의) 방향 결정. 최종 유스케이스의 핵심이지만 가장 큰 작업이라 장기 과제.
8. **UI**: Streamlit/Gradio. 모드 선택 드롭다운, 아이템 아이콘+명칭 카드(아이콘 URL은 `type=item` 레코드의 `icon`).
9. 최종 단계: 타팀 기술 보고서 요약 제출 + 자팀 기술 보고서 작성, 두 트랙 통합 데모.

---

## 7. 검증 필요 항목 / 발표자료 정정 사항

- **토크나이저 관련 표현 정정**: 슬라이드/대본의 "토크나이저 **직접 구현**"은 부정확하다. BPE 알고리즘은 HuggingFace `tokenizers` 라이브러리가 수행하고, 팀이 한 일은 정규화(NFKC)·숫자 한 자리 분리·도메인 용어 단일 토큰 보존·스페셜 토큰 설계. **"도메인 특화 토크나이저 설계 및 학습"** 으로 고쳐 쓸 것 (교수가 "BPE를 직접 짰나?"라고 물으면 곤란).
- **미검증 성과 주장**: 현재 슬라이드 9에 "테스트 시 복원 결과 원문과 100% 일치 확인", "토크나이저 구현 완료"라고 적혀 있으나, **토크나이저를 실제 코퍼스로 학습·테스트한 출력을 아직 받은 적이 없다.** 위 6장 2번을 실행해 `일치: True`를 직접 확인한 뒤에 발표할 것. 실패하면 문구 수정.
- **"트랜스포머 만들었다"**: 모델 코드는 아직 없다. 발표자료에도 토크나이저까지만 완료로 적어둠. 모델 코드가 따로 있으면 사용자가 공유해야 반영 가능.
- 모드 대응표(맵 ID, 주문 modes)는 미검증 (5.2).
- 증강 칼바람/TFT 데이터는 아직 수집 안 됨 (모드 이름만 용어 사전에 포함).

---

## 8. 핵심 코드 (현재 작동 중인 버전)

### 8.1 패치노트 크롤러 (`crawler.py`) — 목록 URL 자동 수집 + 범위 제한 크롤링

```python
import requests
from bs4 import BeautifulSoup
import time
from collections import Counter

def get_patch_note_urls(list_url, max_patches=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(list_url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    patch_urls, seen = [], set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "patch-" in href and "-notes" in href and href not in seen:
            full = href if href.startswith("http") else f"https://www.leagueoflegends.com{href}"
            patch_urls.append(full)
            seen.add(href)
    return patch_urls[:max_patches]

def crawl_patch_notes(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    start = soup.find(id="patch-champions")          # 챔피언 변경사항 섹션 시작점
    if not start:
        print(f"  ⚠️ {url} — patch-champions 영역 없음, 건너뜀")
        return []

    elements = []
    for el in start.find_all_next(["h2", "a", "h4", "li"]):
        if el.name == "h2":                           # 다음 섹션 도달 -> 중단 (볼리베어 귀속 버그 방지)
            break
        elements.append(el)

    data, champion, skill = [], None, None
    patch_version = url.split("patch-")[-1].replace("-notes", "")   # 예: 26-19

    for el in elements:
        if el.name == "a":
            text, href = el.get_text(strip=True), el.get("href", "")
            if text and len(text) <= 15 and ("/champion" in href or "/ko-kr/champions" in href):
                champion, skill = text, None
        elif el.name == "h4":
            text = el.get_text(strip=True)
            if text:
                skill = text
        elif el.name == "li":
            text = el.get_text(strip=True)
            if text and champion:
                data.append({"champion": champion, "skill": skill or "일반",
                             "change": text, "patch": patch_version})
    return data

list_url = "https://www.leagueoflegends.com/ko-kr/news/tags/patch-notes/"
urls = get_patch_note_urls(list_url, max_patches=5)
all_data = []
for url in urls:
    print("크롤링 중:", url)
    d = crawl_patch_notes(url)
    print(f"  → {len(d)}건")
    all_data.extend(d)
    time.sleep(1)

print("패치별:", sorted(Counter(d["patch"] for d in all_data).items()))
with open("crawled_data.txt", "w", encoding="utf-8") as f:
    for d in all_data:
        f.write(f"[patch_{d['patch']}][{d['champion']}] {d['skill']}: {d['change']}\n")
```

파일 형식 예: `[patch_26-19][아트록스] W - 지옥사슬: 재사용 대기시간: 20/18/16/14/12초 ⇒18/16.5/15/13.5/12초`

### 8.2 청킹 (`chunking.py`) — 챔피언×패치 단위

```python
import re
from collections import defaultdict, Counter

pattern = re.compile(r"^\[patch_([^\]]+)\]\[([^\]]+)\]\s*(.+)$")
groups = defaultdict(list)
with open("crawled_data.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            print("⚠️ 파싱 실패:", line[:50]); continue
        patch, champion, rest = m.groups()
        groups[(patch, champion)].append(rest)

chunks = [{"text": f"{c} {p} 패치 변경사항:\n" + "\n".join(v), "champion": c, "patch": p}
          for (p, c), v in groups.items()]
print(f"총 {len(chunks)}개 청크", sorted(Counter(c["patch"] for c in chunks).items()))

with open("chunks.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(f"=== {c['champion']} | patch_{c['patch']} ===\n{c['text']}\n\n")
```

### 8.3 챔피언 기본 데이터 (Data Dragon API)

```python
import requests
ver = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
champs = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/ko_KR/champion.json").json()["data"]
docs = []
for key in champs:
    d = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/ko_KR/champion/{key}.json").json()["data"][key]
    s, p = d["stats"], d["passive"]
    lines = [f"{d['name']} ({d['title']})", f"역할군: {', '.join(d['tags'])}",
             f"기본 체력: {s['hp']}, 공격력: {s['attackdamage']}, 방어력: {s['armor']}, 이동속도: {s['movespeed']}",
             f"기본 지속 효과 - {p['name']}: {p['description']}"]
    for k, sp in zip("QWER", d["spells"]):
        lines.append(f"{k} - {sp['name']}: {sp['description']}")
    docs.append((d["name"], "\n".join(lines)))
with open("champion_base_data.txt", "w", encoding="utf-8") as f:
    for name, text in docs:
        f.write(f"=== {name} ===\n{text}\n\n")
```

### 8.4 Vector DB 구축 + 하이브리드 검색 (`vectorDB.py`)

**주의**: 패치 청크 적재 코드는 컬렉션을 삭제 후 재생성하므로, **반드시 먼저 실행하고 그다음에 기본정보를 추가**할 것.

```python
import re
import chromadb
from chromadb.utils import embedding_functions

MODEL = "distiluse-base-multilingual-cased-v1"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)
client = chromadb.PersistentClient(path="./chroma_db")

# --- (1) 패치 청크 적재: 컬렉션 재생성 ---
try:
    client.delete_collection("gamehikking")
except Exception:
    pass
collection = client.get_or_create_collection(name="gamehikking", embedding_function=ef)

header_re = re.compile(r"^(.+?) \| patch_(.+?) ===\n(.+)$", re.DOTALL)
patch_chunks = []
for block in open("chunks.txt", encoding="utf-8").read().strip().split("=== ")[1:]:
    m = header_re.match(block)
    if m:
        patch_chunks.append({"champion": m.group(1).strip(), "patch": m.group(2).strip(), "text": m.group(3).strip()})
for i, c in enumerate(patch_chunks):
    collection.add(documents=[c["text"]], metadatas=[{"champion": c["champion"], "patch": c["patch"]}], ids=[f"chunk_{i}"])

# --- (2) 챔피언 기본정보 추가 ---
for i, block in enumerate(open("champion_base_data.txt", encoding="utf-8").read().strip().split("=== ")[1:]):
    champion, text = block.split(" ===\n", 1)
    text = re.sub(r"<br\s*/?>", " ", text.strip())
    collection.add(documents=[text], metadatas=[{"champion": champion.strip(), "type": "base_info", "patch": "N/A"}], ids=[f"base_{i}"])
print("DB 총 청크:", collection.count())   # 237

# --- (3) 하이브리드 검색: 챔피언명/패치번호 감지 -> 메타데이터 필터 + 벡터 검색 ---
known_champions = sorted({m["champion"] for m in collection.get()["metadatas"] if m.get("champion")}, key=len, reverse=True)

def smart_search(query, n_results=3):
    champ = next((c for c in known_champions if c in query), None)
    pm = re.search(r"(\d{2}-\d{2})", query)
    patch = pm.group(1) if pm else None
    where = None
    if champ and patch:
        where = {"$and": [{"champion": champ}, {"patch": patch}]}
    elif champ:
        where = {"champion": champ}
    return collection.query(query_texts=[query], n_results=n_results, where=where), champ, patch
```

향후 모드 필터 추가 시: `where`에 `{"mode": {"$in": [선택모드, "공통"]}}`를 `$and`로 결합.

### 8.5 토크나이저 (`lol_tokenizer.py`, 사용자가 보유) — 요약

- 서브커맨드: `train`(`--data` 파일/폴더 `.jsonl/.json/.txt`, `--terms`, `--vocab-size`, `--min-frequency`, `--out`), `test`(`--tok`, `--text`).
- 설계: **Byte-level BPE**(UNK 방지) / **NFKC 정규화**(대소문자 유지: Q/W/E/R 보존) / **숫자 한 자리씩 분리**(`Digits(individual_digits=True)`) / **도메인 용어를 `AddedToken(special=False)`로 추가해 단일 토큰 보존** / HF `PreTrainedTokenizerFast`로 래핑해 `save_pretrained`.
- 스페셜 토큰: `<pad> <unk> <s> </s>`, 채팅용 `<|system|> <|user|> <|assistant|> <|end|>`, 메타용 `<|champion|> <|role|> <|patch|> <|section|>`. `format_chat(messages)` 헬퍼 포함.
- 모델 학습 코드에서: `from lol_tokenizer import load; tok = load('./tokenizer_out')`.
- 알려진 주의: 도메인 용어를 `add_tokens`로 추가하면 vocab 크기가 `--vocab-size`보다 커진다(용어 개수만큼 증가). 모델 임베딩 크기는 `len(tok)` 기준으로 잡을 것.

---

## 9. 발표 자료 현황

- 파일: `GameHikking_week4.pptx` / `.pdf` (10슬라이드), `GameHikking_발표대본_v3.md`. 디자인: 남색(0A1428)+골드(C8AA6E)+틸(0AC8B9), 제목 Cambria / 본문 Calibri, 16:9. pptxgenjs로 생성했으며 빌드 스크립트는 이전 AI의 샌드박스에 있어 **새 환경에서는 사라졌을 수 있음** → 필요하면 아래 구성으로 재생성.
- 슬라이드 구성: ① 타이틀 ② 선정 주제(리캡) ③ 이번 주 목표 ④ 챔피언 귀속 오류 해결(before/after) ⑤ 여러 페이지 자동 크롤링(5패치·145건) ⑥ 챔피언 기본 데이터셋(API·173명) ⑦ 임베딩·검색 품질 실험(문제 2건) ⑧ 하이브리드 검색 최종 테스트(5/5) ⑨ 연구 트랙: 토크나이저 ⑩ 진행상황 종합·다음 계획.
- 이전에 정한 발표 톤: "문제 발견 → 원인 분석 → 해결 → 검증" 스토리로 풀어서 기술적 깊이를 보여주기. 크롤링/청킹/임베딩/Vector DB/RAG 같은 기술 개념은 "무엇인가 + 왜 쓰는가"를 대본에 설명 박스로 포함.
- **다음 발표 때 갱신할 것**: 7장의 토크나이저 표현 정정, 모드 태깅/아이템 데이터 수집 결과, 토크나이저 학습·비교 수치, 미니 트랜스포머 진행, 최종 유스케이스 방향.

---

## 10. 폴더 구조 (예상)

```
GameHikking/
├─ crawler.py / preCrawler.py     # 패치노트 크롤러, URL 수집 테스트
├─ chunking.py                    # 챔피언×패치 청킹
├─ vectorDB.py                    # Chroma 적재 + 하이브리드 검색
├─ make_corpus.py                 # (5.5, 실행 여부 확인)
├─ collect_lol_data.py            # (5.3, 미실행)
├─ lol_tokenizer.py               # 토크나이저 (사용자 보유)
├─ crawled_data.txt  chunks.txt  champion_base_data.txt
├─ chroma_db/                     # Vector DB (distiluse 임베딩)
├─ data/
│  ├─ lol_terms.txt               # 도메인 용어 사전
│  └─ chunks/  corpus.txt, ddragon_extra.jsonl
└─ tokenizer_out/                 # 토크나이저 학습 결과 (미생성)
```

패키지: `requests beautifulsoup4 chromadb sentence-transformers tokenizers transformers` (미니 트랜스포머용 `torch`는 아직 미설치).
