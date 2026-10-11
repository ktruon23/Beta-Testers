#!/usr/bin/env bash
# Start all Beta Agents services in the background; Ctrl+C stops them all.
cd "$(dirname "$0")"
trap 'kill $(jobs -p) 2>/dev/null' EXIT

python -m uvicorn main:app         --app-dir vision-agent  --port 8001 --env-file .env &
python -m uvicorn planner:app      --app-dir planner-agent --port 8002 --env-file .env &
python -m uvicorn critic:app       --app-dir planner-agent --port 8003 --env-file .env &
python -m uvicorn orchestrator:app --app-dir planner-agent --port 8000 --env-file .env &
python -m uvicorn coach:app        --app-dir coach-agent   --port 8004 --env-file .env &

wait
