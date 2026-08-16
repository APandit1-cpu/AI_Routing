#!/usr/bin/env bash
# start.sh - run the AI Routing app in the background
#
# Usage: ./start.sh   (from the project root)

set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PID_FILE="app.pid"
LOG_FILE="app.log"
PORT=5050

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "App is already running (PID $(cat "$PID_FILE")) on port $PORT."
  exit 0
fi

# Create venv + install deps if needed
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  uv venv "$VENV"
fi
if ! "$VENV/bin/python" -c "import flask, torch, sklearn" 2>/dev/null; then
  echo "Installing dependencies..."
  uv pip install -r requirements.txt --python "$VENV/bin/python"
fi

echo "Starting app in background..."
nohup "$VENV/bin/python" app.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "App started (PID $(cat "$PID_FILE")). Logs: $LOG_FILE. http://localhost:$PORT"
