# Beta Testers

A bouldering route planner built from cooperating (and sometimes disagreeing) AI agents. Upload a photo of a climbing wall, enter your height, wingspan and level, and get a route you can actually reach, with move-by-move beta and a difficulty grade.

Built for the Cisco "When AI Agents Disagree" track.

## How it works

Five FastAPI services talk to each other over HTTP using the shared schemas in `backend/shared/schemas.py`:

| Agent        | Port | Job |
|--------------|------|-----|
| Orchestrator | 8000 | Runs the conversation: sends the photo to Vision, asks the Planner for a route, gets the Critic's verdict, re-plans, and hands the final route to the Coach. Streams every message to the page. |
| Vision       | 8001 | Finds holds in the photo (Roboflow), their colors, and the wall's scale in inches per pixel. |
| Planner      | 8002 | Finds the cheapest reachable path with Dijkstra. Every move must fit within your reach (wingspan x 0.85). Claude picks a strategy (safe, balanced, efficient) and compares candidates. Python computes all the geometry. |
| Critic       | 8003 | Reviews each plan. It can reject holds or moves that are too far for you or land on bad holds, and it grades the final route (V0-V10). |
| Coach        | 8004 | Writes move-by-move instructions and tips for the chosen route. |

**The disagreement loop:** the Planner proposes and the Critic approves or pushes back with holds and moves to avoid. The Planner then re-plans and can push back itself, with numbers, when the constraints make the route much worse. This repeats for a few rounds. If the Critic never approves, the Orchestrator keeps the best plan it has.

Climber level changes the search. Beginners get shorter moves (at most 80% of reach) and extra penalties for crimps, slopers and pockets. Experts favor fewer moves.

## Repo layout

```
backend/
  planner-agent/   orchestrator.py, planner.py, critic.py
  vision-agent/    hold detection, colors, scale
  coach-agent/     beta writer (+ SQLite feedback history)
  shared/          schemas.py, llm.py (Claude / OpenAI-compatible client)
  agent-viewer/    Streamlit app that shows each agent step live
  run_agents.sh    starts all five services
frontend/          React + Vite page
tests/             pytest suite (runs every agent in-process, no network)
pytest.ini
```

## Setup

Requires Python 3.10+ and Node 20+.

1. Create `backend/.env` from the template and fill in your keys:

   ```
   cp backend/.env.example backend/.env
   ```

   `ANTHROPIC_API_KEY` is optional; it enables Claude in every agent. The Roboflow keys (`ROBOFLOW_API_KEY`, `WORKSPACE_NAME`, `WORKFLOW_ID`) are needed for hold detection. To use an OpenAI-compatible server (for example NVIDIA Brev or vLLM) instead of Claude, uncomment the `LLM_*` lines in the template.

2. Install and start the agents from the repo root:

   ```
   pip install -r backend/requirements.txt
   bash backend/run_agents.sh
   ```

3. Start the frontend:

   ```
   cd frontend
   npm install
   npm run dev
   ```

   Open http://localhost:5173. The dev server proxies API calls to the Orchestrator on port 8000. Set `ORCHESTRATOR_URL` to point it somewhere else.

Optional live view of the agents: `streamlit run backend/agent-viewer/app.py`.

## Tests

```
pip install -r backend/requirements-dev.txt
python -m pytest
```

Run this from the repo root. The suite wires the agents together in-process and needs no API keys. GitHub Actions runs it, plus the frontend build, on every push and pull request (`.github/workflows/ci.yml`).

## Without an API key

Nothing breaks. With no LLM configured, or if a Claude call fails (bad key, network, rate limit), each agent falls back to deterministic logic:

- The Orchestrator runs a fixed plan-critique loop.
- The Planner uses the default strategy for your level.
- The Critic uses reach and hold-type rules.
- The Coach writes templated beta.

The page labels which path was used, for example "fixed pipeline (no API key)".

## Limitations

- The difficulty grade is a rule-based estimate from hold types and reach. It has not been calibrated against real gym grades.
- The planner only models hand moves. It ignores feet, body position, matching and dynos, and it minimizes total move cost rather than the single hardest move.
- The scale is approximate. It is estimated from the median hold size unless you set it by tapping two points and entering the real distance. The reach factor (0.85 x wingspan) is also a guess.
- Hold types come from the camera or from your taps. Unknown holds are assumed to be a default type, and the grade says so.

## License

MIT. See [LICENSE](LICENSE).
