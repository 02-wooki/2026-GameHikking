#!/usr/bin/env bash
# 맥 -> 라즈베리파이 배포. 프로젝트 루트에서 실행: bash deploy/deploy_pi.sh
#   1) 코드 최신화 (git pull)  2) git에 없는 데이터 rsync  3) 의존성 확인  4) systemd 재시작
# 환경변수로 덮어쓰기: PI=pi-out PI_DIR=/home/sysadmin/2026-GameHikking
set -euo pipefail

PI="${PI:-pi-out}"
PI_DIR="${PI_DIR:-/home/sysadmin/2026-GameHikking}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1. 파이 코드 최신화"
ssh "$PI" "cd '$PI_DIR' && git pull --ff-only"

echo "== 2. 데이터 동기화 (chroma_db, data/processed, ddragon champion)"
ssh "$PI" "mkdir -p '$PI_DIR/data/processed' '$PI_DIR/data/raw/ddragon/16.22.1/champion'"
rsync -az --delete --stats chroma_db/ "$PI:$PI_DIR/chroma_db/"
rsync -az --stats data/processed/ "$PI:$PI_DIR/data/processed/"
rsync -az --stats data/raw/ddragon/16.22.1/champion/ "$PI:$PI_DIR/data/raw/ddragon/16.22.1/champion/"

echo "== 3. 가상환경 의존성 확인"
ssh "$PI" "cd '$PI_DIR' && .venv/bin/python -c 'import torch, chromadb, sentence_transformers, streamlit, requests, bs4; print(\"deps OK\")' \
  || (.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      .venv/bin/pip install chromadb sentence-transformers streamlit requests beautifulsoup4)"

echo "== 4. systemd 등록/재시작"
scp deploy/gamehikking.service "$PI:/tmp/gamehikking.service"
ssh -t "$PI" "sudo mv /tmp/gamehikking.service /etc/systemd/system/gamehikking.service && \
  sudo systemctl daemon-reload && sudo systemctl enable --now gamehikking && sudo systemctl restart gamehikking"

echo "== 5. 상태 확인 (첫 실행은 임베딩 모델 ~500MB 다운로드로 느림)"
ssh "$PI" "sleep 8; systemctl --no-pager -l status gamehikking | head -15; curl -s -o /dev/null -w 'localhost:8510 -> HTTP %{http_code}\n' http://127.0.0.1:8510 || true"

echo
echo "공개 주소: https://game.wooki.cloud (CloudPanel 리버스 프록시 -> 127.0.0.1:8510). 첫 접속은 모델 로딩으로 ~20초"
