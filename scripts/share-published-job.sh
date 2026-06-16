#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-ff4ad73f66e2}"
PORT="${2:-8877}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLISH_DIR="$ROOT/data/publish/$JOB_ID"

if [[ ! -d "$PUBLISH_DIR" ]]; then
  echo "Missing publish folder. Run: interview-mapper publish $JOB_ID"
  exit 1
fi

cd "$PUBLISH_DIR"
echo "Serving $PUBLISH_DIR on http://127.0.0.1:$PORT"
python3 -m http.server "$PORT" --bind 127.0.0.1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 1
echo "Opening public tunnel (keep this terminal open)..."
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -R "80:127.0.0.1:$PORT" nokey@localhost.run
