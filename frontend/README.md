# Beta Agents frontend

React + Vite page for the Beta Agents route planner. It started from the Figma Make design and now calls the orchestrator for real results.

## Run it

1. Start the agents from `backend/`: `bash run_agents.sh` (the orchestrator must be on port 8000).
2. In this folder:

```
npm install
npm run dev
```

3. Open http://localhost:5173

The page calls `/api/...`, and Vite forwards that to `http://localhost:8000`, so the backend needs no CORS setup.
To use an orchestrator somewhere else, set `ORCHESTRATOR_URL=http://host:8000` before `npm run dev`.

## What it calls

- `GET /health/all` when the page loads, to show a banner if an agent is down.
- `POST /detect` right after a photo is added. This previews the holds and fills the route color list.
- `POST /solve_photo` on "Find a route", with the climber's height, wingspan, level and the chosen color.

Photos are shrunk to 1600 px on the long side before upload. Hold coordinates come back in that image's pixels, which is the same image the page draws on.

The Planner's tries are read from the `trace` (each Planner entry carries `payload.path`), and they become the "Try 1 / Final" tabs. Holds the Critic told the Planner to avoid are drawn with an X.
