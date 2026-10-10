"""
ORCHESTRATOR  -- no LLM
The only service the frontend talks to. It routes messages between the other
agents and records every message in a `trace` so the demo can show the agents
talking to each other.

The flow:
    Vision  -> hold map              (only if the user asks for auto-detect)
    Planner -> route
    Critic  -> approve, or reject with constraints --+
    Planner -> new route with constraints  <---------+   (repeat up to max_rounds)
    Coach   -> move-by-move beta
"""

import os

import httpx
from fastapi import FastAPI, HTTPException

from shared.schemas import (
    CoachResponse,
    CritiqueResponse,
    DetectRequest,
    DetectResponse,
    PlanResponse,
    SolveRequest,
    SolveResponse,
    TraceEntry,
)

VISION_URL = os.environ.get("VISION_URL", "http://localhost:8001")
PLANNER_URL = os.environ.get("PLANNER_URL", "http://localhost:8002")
CRITIC_URL = os.environ.get("CRITIC_URL", "http://localhost:8003")
COACH_URL = os.environ.get("COACH_URL", "http://localhost:8004")

TIMEOUT = httpx.Timeout(90.0)  # LLM calls + free-tier cold starts can be slow

app = FastAPI(title="Orchestrator")


def send(client: httpx.Client, url: str, body: dict) -> dict:
    try:
        r = client.post(url, json=body)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Agent at {url} failed: {e}") from e


@app.get("/")
def health():
    return {"agent": "orchestrator", "status": "ok"}


@app.get("/health/all")
def health_all():
    """Ping every agent. Hit this right before a demo to wake sleeping services."""
    out = {}
    with httpx.Client(timeout=TIMEOUT) as client:
        for name, url in [("vision", VISION_URL), ("planner", PLANNER_URL), ("critic", CRITIC_URL), ("coach", COACH_URL)]:
            try:
                out[name] = client.get(url + "/").json()
            except Exception as e:
                out[name] = {"status": "DOWN", "error": str(e)}
    return out


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    with httpx.Client(timeout=TIMEOUT) as client:
        return send(client, f"{VISION_URL}/detect", req.model_dump())


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    trace: list[TraceEntry] = []

    def log(sender, receiver, summary, payload=None):
        trace.append(TraceEntry(step=len(trace) + 1, sender=sender, receiver=receiver, summary=summary, payload=payload or {}))

    if len(req.holds) < 2:
        raise HTTPException(400, "Need at least 2 holds.")

    ids = {h.id for h in req.holds}
    # Default start = lowest hold on the wall, finish = highest (image y grows down)
    start_id = req.start_id if req.start_id in ids else max(req.holds, key=lambda h: h.y).id
    finish_id = req.finish_id if req.finish_id in ids else min(req.holds, key=lambda h: h.y).id

    avoid_holds: set[int] = set()
    avoid_moves: list[list[int]] = []
    critic_feedback: dict | None = None
    last_good: PlanResponse | None = None
    approved = False
    rounds = 0
    status = ""

    with httpx.Client(timeout=TIMEOUT) as client:
        for rnd in range(1, req.max_rounds + 1):
            rounds = rnd
            plan_req = {
                "holds": [h.model_dump() for h in req.holds],
                "start_id": start_id,
                "finish_id": finish_id,
                "inches_per_pixel": req.inches_per_pixel,
                "climber": req.climber.model_dump(),
                "avoid_holds": sorted(avoid_holds),
                "avoid_moves": avoid_moves,
                "critic_feedback": critic_feedback,
            }
            constraint_note = (
                f" with constraints: avoid holds {sorted(avoid_holds)}, avoid moves {avoid_moves}"
                if (avoid_holds or avoid_moves) else ""
            )
            log("Orchestrator", "Planner", f"Round {rnd}: find a route #{start_id} -> #{finish_id}{constraint_note}")
            plan = PlanResponse(**send(client, f"{PLANNER_URL}/plan", plan_req))

            if not plan.ok:
                log("Planner", "Orchestrator", plan.error or "No route.")
                if last_good is None:
                    return SolveResponse(
                        ok=False, status=plan.error or "No route found.", holds=req.holds,
                        start_id=start_id, finish_id=finish_id, plan=plan, trace=trace, rounds=rounds,
                    )
                status = "The Critic's constraints left no route, so we kept the previous one."
                break

            last_good = plan
            path_str = " -> ".join(f"#{p}" for p in plan.path)
            worst = max(m.reach_ratio for m in plan.moves)
            log("Planner", "Critic", f"Proposed {len(plan.moves)} moves: {path_str} (hardest move {worst:.0%} of reach)",
                {"path": plan.path})

            crit = CritiqueResponse(**send(client, f"{CRITIC_URL}/critique", {
                "plan": plan.model_dump(),
                "holds": [h.model_dump() for h in req.holds],
                "climber": req.climber.model_dump(),
                "start_id": start_id,
                "finish_id": finish_id,
                "round": rnd,
                "max_rounds": req.max_rounds,
                "planner_pushback": plan.pushback.model_dump() if plan.pushback else None,
            }))
            verdict = "APPROVED" if crit.approved else "REJECTED"
            log("Critic", "Planner", f"{verdict}: {crit.reasoning}",
                {"avoid_holds": crit.avoid_holds, "avoid_moves": crit.avoid_moves, "source": crit.source})

            # Feed the Critic's last verdict into the next Planner round so it can
            # respond to the actual reasoning, not only the accumulated constraints.
            critic_feedback = {
                "reasoning": crit.reasoning,
                "avoid_holds": crit.avoid_holds,
                "avoid_moves": crit.avoid_moves,
            }

            if crit.approved:
                approved = True
                status = f"Route approved by the Critic in round {rnd}."
                break

            new_holds = set(crit.avoid_holds) - {start_id, finish_id} - avoid_holds
            new_moves = [m for m in crit.avoid_moves if m not in avoid_moves]
            if not new_holds and not new_moves:
                status = "The Critic rejected the route but gave nothing new to change, so we kept it."
                break
            avoid_holds |= new_holds
            avoid_moves += new_moves
        else:
            status = f"Hit the {req.max_rounds}-round limit; using the last route."

        log("Orchestrator", "Coach", "Write move-by-move beta for the final route.")
        beta = CoachResponse(**send(client, f"{COACH_URL}/beta", {
            "plan": last_good.model_dump(),
            "holds": [h.model_dump() for h in req.holds],
            "climber": req.climber.model_dump(),
        }))
        log("Coach", "Orchestrator", beta.summary, {"source": beta.source})

    return SolveResponse(
        ok=True, status=status, holds=req.holds, start_id=start_id, finish_id=finish_id,
        plan=last_good, critic_approved=approved, beta=beta, trace=trace, rounds=rounds,
    )
