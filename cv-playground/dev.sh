#!/usr/bin/env bash
# Chạy cả backend (FastAPI :8000) và frontend (Vite :5173) bằng 1 lệnh.
# Dùng: ./dev.sh
set -euo pipefail
cd "$(dirname "$0")"

# Backend venv
if [ ! -x .venv/bin/uvicorn ]; then
  echo "[dev] creating .venv and installing requirements..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

# Frontend deps
if [ ! -d frontend/node_modules ]; then
  echo "[dev] installing frontend dependencies..."
  (cd frontend && npm install)
fi

PIDS=()
cleanup() {
  echo
  echo "[dev] shutting down..."
  kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# `exec` để PID bắt được chính là server, kill sạch khi Ctrl+C
(cd backend && exec ../.venv/bin/uvicorn app.main:app --reload --port 8000) &
PIDS+=($!)

(cd frontend && exec ./node_modules/.bin/vite) &
PIDS+=($!)

echo "[dev] backend  -> http://localhost:8000  (docs: /docs)"
echo "[dev] frontend -> http://localhost:5173"
echo "[dev] Ctrl+C to stop both"

wait
