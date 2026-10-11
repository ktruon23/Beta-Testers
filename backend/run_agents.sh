#!/usr/bin/env bash
# Start all Beta Testers services in the background; Ctrl+C stops them all.
# Runs from the repo root (parent of backend/) so `backend.shared` is importable.
cd "$(dirname "$0")/.."
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# A stale SSL_CERT_FILE (e.g. left by a conda env) makes every httpx client crash; fall back to the default certs.
if [ -n "$SSL_CERT_FILE" ] && [ ! -f "$SSL_CERT_FILE" ]; then
  echo "Ignoring SSL_CERT_FILE=$SSL_CERT_FILE (file not found)"
  unset SSL_CERT_FILE
fi

python -m uvicorn main:app         --app-dir backend/vision-agent  --port 8001 --env-file backend/.env &
python -m uvicorn planner:app      --app-dir backend/planner-agent --port 8002 --env-file backend/.env &
python -m uvicorn critic:app       --app-dir backend/planner-agent --port 8003 --env-file backend/.env &
python -m uvicorn orchestrator:app --app-dir backend/planner-agent --port 8000 --env-file backend/.env &
python -m uvicorn coach:app        --app-dir backend/coach-agent   --port 8004 --env-file backend/.env &

wait
