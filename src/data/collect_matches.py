# src/data/collect_matches.py
"""A안: Riot API로 한국 서버 상위 랭크(챌린저·그랜드마스터·마스터) 솔로랭크 매치 수집 (백그라운드용)
- 결과: data/raw/riot/matches/<매치ID>.json (개인 식별 정보 제거한 원본)
         data/raw/riot/participants.jsonl (분석용 요약: 챔피언, 포지션, 상대 라이너, 최종 아이템, 승패, 버전)
- 이어받기: 이미 받은 매치는 건너뛴다. 중단(Ctrl+C) 후 다시 실행하면 이어서 수집.
- 속도: 개발용 키 제한(2분 100회)에 맞춰 약 1.3초 간격. 429 응답이면 Retry-After만큼 대기.
- 개인정보: 소환사명·Riot ID·puuid 등은 저장하지 않는다.

준비:
  1) https://developer.riotgames.com 에 Riot 계정으로 로그인 -> Development API Key 발급 (24시간마다 만료)
  2) 프로젝트 루트 .env 파일에 한 줄:  RIOT_API_KEY=RGAPI-xxxxxxxx   (.env는 .gitignore에 있음)
실행 (프로젝트 루트에서):
  caffeinate -dims /usr/local/bin/python3.10 src/data/collect_matches.py --max 3000 --days 14
  (--timeline 을 붙이면 아이템 구매 순서용 타임라인도 받음: 요청 수 2배)
  (--no-raw: 원본 매치 JSON 저장 안 함, 요약 participants.jsonl만 -> 용량 대폭 절약)
  (--seed-start N: 시드 플레이어 N번째부터 시작)
이어받기 기준은 participants.jsonl의 매치 ID다. 다른 컴퓨터에서 이어서 받으려면
participants.jsonl 하나만 같은 경로(data/raw/riot/)에 복사해 두고 실행하면 된다.
※ 같은 API 키로 두 컴퓨터에서 동시에 돌리면 요청 제한(429)에 걸리니 한 곳에서만 실행할 것
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "riot"
MATCH_DIR = OUT / "matches"
TL_DIR = OUT / "timelines"
SUMMARY = OUT / "participants.jsonl"
SEEDS = OUT / "seed_puuids.json"
PLATFORM = "https://kr.api.riotgames.com"
REGION = "https://asia.api.riotgames.com"
QUEUE = 420                                   # 솔로 랭크
INTERVAL = 1.3                                # 초 (2분 100회 제한 대응)
DROP = {"riotIdGameName", "riotIdTagline", "summonerName", "summonerId", "puuid", "profileIcon", "summonerLevel"}


def load_key():
    key = os.environ.get("RIOT_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("RIOT_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("RIOT_API_KEY가 없습니다. 프로젝트 루트 .env에 RIOT_API_KEY=RGAPI-... 를 넣어 주세요.")
    return key


S = requests.Session()
_last = [0.0]


def get(url, **params):
    for attempt in range(6):
        wait = INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = S.get(url, params=params, timeout=15)
        except requests.RequestException as e:
            print(f"  네트워크 오류, 재시도: {e}")
            time.sleep(5)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            sec = int(r.headers.get("Retry-After", 10))
            print(f"  요청 제한(429): {sec}초 대기")
            time.sleep(sec)
            continue
        if r.status_code in (401, 403):
            sys.exit(f"API 키 오류({r.status_code}): 키가 만료됐거나 잘못됐습니다. 새 키를 .env에 넣고 다시 실행하세요. "
                     f"(이미 받은 매치는 유지되고 이어서 수집됩니다)")
        if r.status_code == 404:
            return None
        print(f"  서버 오류 {r.status_code}, 재시도")
        time.sleep(5)
    return None


def seed_puuids():
    if SEEDS.exists():
        return json.loads(SEEDS.read_text(encoding="utf-8"))
    puuids = []
    for tier in ("challengerleagues", "grandmasterleagues", "masterleagues"):
        data = get(f"{PLATFORM}/lol/league/v4/{tier}/by-queue/RANKED_SOLO_5x5")
        entries = (data or {}).get("entries", [])
        got = 0
        for e in entries:
            if e.get("puuid"):
                puuids.append(e["puuid"])
                got += 1
            elif e.get("summonerId"):                         # 구버전 응답 대비
                s = get(f"{PLATFORM}/lol/summoner/v4/summoners/{e['summonerId']}")
                if s and s.get("puuid"):
                    puuids.append(s["puuid"])
                    got += 1
        print(f"  {tier}: {got}명")
    random.Random(0).shuffle(puuids)
    OUT.mkdir(parents=True, exist_ok=True)
    SEEDS.write_text(json.dumps(puuids), encoding="utf-8")
    return puuids


def summarize(m):
    info = m["info"]
    ps = info["participants"]
    rows = []
    for p in ps:
        opp = next((o for o in ps if o["teamId"] != p["teamId"] and o.get("teamPosition") == p.get("teamPosition")
                    and p.get("teamPosition")), None)
        rows.append({
            "match_id": m["metadata"]["matchId"], "version": info.get("gameVersion"),
            "duration": info.get("gameDuration"), "champion": p.get("championName"), "champion_id": p.get("championId"),
            "position": p.get("teamPosition"), "team": p.get("teamId"), "win": p.get("win"),
            "opponent": opp.get("championName") if opp else None,
            "items": [p.get(f"item{i}") for i in range(7)],
            "spells": [p.get("summoner1Id"), p.get("summoner2Id")],
            "kda": [p.get("kills"), p.get("deaths"), p.get("assists")],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3000, help="이번 실행에서 받을 최대 매치 수")
    ap.add_argument("--days", type=int, default=14, help="최근 며칠 매치만")
    ap.add_argument("--timeline", action="store_true", help="타임라인(구매 순서)도 받기")
    ap.add_argument("--no-raw", action="store_true", help="원본 매치 JSON 저장 안 함 (요약만)")
    ap.add_argument("--seed-start", type=int, default=0, help="시드 플레이어 시작 번호")
    a = ap.parse_args()

    S.headers["X-Riot-Token"] = load_key()
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    TL_DIR.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in MATCH_DIR.glob("*.json")}
    if SUMMARY.exists():                                   # 요약 파일 기준으로 이어받기 (다른 컴퓨터와 공유 가능)
        with SUMMARY.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    have.add(json.loads(line)["match_id"])
    print(f"기존 매치 {len(have)}개 | 시드 수집 중...")
    puuids = seed_puuids()
    print(f"시드 플레이어 {len(puuids)}명")
    start_time = int(time.time()) - a.days * 86400

    new, t0 = 0, time.time()
    with SUMMARY.open("a", encoding="utf-8") as fs:
        for k, pu in enumerate(puuids):
            if k < a.seed_start:
                continue
            ids = get(f"{REGION}/lol/match/v5/matches/by-puuid/{pu}/ids",
                      queue=QUEUE, type="ranked", startTime=start_time, count=20) or []
            for mid in ids:
                if mid in have:
                    continue
                m = get(f"{REGION}/lol/match/v5/matches/{mid}")
                if not m:
                    continue
                for p in m["info"]["participants"]:
                    for key in DROP:
                        p.pop(key, None)
                m["metadata"].pop("participants", None)              # puuid 목록 제거
                if not a.no_raw:
                    (MATCH_DIR / f"{mid}.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
                for row in summarize(m):
                    fs.write(json.dumps(row, ensure_ascii=False) + "\n")
                fs.flush()
                if a.timeline:
                    tl = get(f"{REGION}/lol/match/v5/matches/{mid}/timeline")
                    if tl:
                        tl["metadata"].pop("participants", None)
                        for p in tl["info"].get("participants", []):
                            p.pop("puuid", None)
                        (TL_DIR / f"{mid}.json").write_text(json.dumps(tl), encoding="utf-8")
                have.add(mid)
                new += 1
                if new % 25 == 0:
                    print(f"  새 매치 {new}개 (총 {len(have)}) | 시드 {k + 1}/{len(puuids)} | {(time.time() - t0) / 60:.1f}분")
                if new >= a.max:
                    print(f"목표 {a.max}개 도달. 종료 (총 {len(have)}개)")
                    return
    print(f"시드 전부 처리. 새 매치 {new}개 (총 {len(have)}개)")


if __name__ == "__main__":
    main()