#!/usr/bin/env bash
# 파이의 질문 로그를 맥으로 가져온다 (분석 + 백업). 프로젝트 루트에서 실행: bash deploy/pull_logs.sh
#   파이: /home/sysadmin/gamehikking-logs/  ->  맥: logs/pi/   (logs/는 gitignore, 커밋 금지)
#   --delete 없이 받으므로 맥 쪽 사본은 지워지지 않고, 파이 파일이 늘어난 만큼만 갱신된다.
# 받은 뒤 요약:  GH_LOG_DIR=logs/pi /usr/local/bin/python3.10 src/rag/query_log.py
set -euo pipefail

PI="${PI:-pi-out}"
PI_LOG_DIR="${PI_LOG_DIR:-/home/sysadmin/gamehikking-logs}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs/pi
rsync -az --stats "$PI:$PI_LOG_DIR/" logs/pi/
# 날짜별 백업 사본도 하나 남겨 둔다 (파이 SD 카드 고장 대비)
mkdir -p logs/pi_backup
for f in logs/pi/*.jsonl; do
  if [ -f "$f" ]; then cp "$f" "logs/pi_backup/$(basename "$f" .jsonl).$(date +%Y%m%d).jsonl"; fi
done
echo
wc -l logs/pi/*.jsonl 2>/dev/null || echo "아직 로그가 없습니다."
echo "요약: GH_LOG_DIR=logs/pi /usr/local/bin/python3.10 src/rag/query_log.py"
