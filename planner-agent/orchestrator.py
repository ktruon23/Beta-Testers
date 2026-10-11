"""
ORCHESTRATOR  -- uses Claude
The only service the frontend talks to. It routes messages between the other
agents and records every message in a `trace` so the demo can show the agents
talking to each other.

With an API key, Claude runs the team. It gets three tools -- plan_route,
critique_route, write_beta -- and decides what to do next: which of the
Critic's constraints to pass on, when to drop one that leaves no route, and
which plan is good enough to coach. Every tool call is logged with Claude's
stated reason.

Claude still never does geometry: plan_route only calls the Planner, which
does all the math. The code also enforces the rules Claude can't break:
start/finish holds are never avoided, the Planner and Critic each run at most
max_rounds times, and the Coach always runs at the end.

Without an API key (or if Claude fails), the fixed pipeline runs instead:
    Planner -> route
    Critic  -> approve, or reject with constraints --+
    Planner -> new route with constraints  <---------+   (repeat up to max_rounds)
    Coach   -> move-by-move beta
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()  # reads ANTHROPIC_API_KEY (and the *_URL overrides) from .env

# Make the repo root importable so `shared` works however this file is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.llm import MODEL, backend, llm_available  # noqa: E402
from shared.schemas import (  # noqa: E402
    CoachResponse,
    CritiqueResponse,
    DetectRequest,
    DetectResponse,
    Hold,
    PlanResponse,
    SolvePhotoRequest,
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
    return {"agent": "orchestrator", "status": "ok", "llm": llm_available()}


@app.get("/health/all")
def health_all():
    """Ping every agent. Hit this right before a demo to wake sleeping services."""
    out = {"orchestrator": health()}
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


# ---------------- calls to the other agents (shared by both modes) ----------------

class Run:
    """One /solve request: the agents' HTTP client plus the trace being recorded."""

    def __init__(self, client: httpx.Client, req: SolveRequest, start_id: int, finish_id: int):
        self.client, self.req = client, req
        self.start_id, self.finish_id = start_id, finish_id
        self.trace: list[TraceEntry] = []
        self.critic_feedback: dict | None = None  # last verdict, so the Planner can push back on it

    def log(self, sender, receiver, summary, payload=None):
        self.trace.append(TraceEntry(step=len(self.trace) + 1, sender=sender, receiver=receiver,
                                     summary=summary, payload=payload or {}))

    def plan(self, avoid_holds: set[int], avoid_moves: list[list[int]]) -> PlanResponse:
        plan = PlanResponse(**send(self.client, f"{PLANNER_URL}/plan", {
            "holds": [h.model_dump() for h in self.req.holds],
            "start_id": self.start_id,
            "finish_id": self.finish_id,
            "inches_per_pixel": self.req.inches_per_pixel,
            "climber": self.req.climber.model_dump(),
            "avoid_holds": sorted(avoid_holds - {self.start_id, self.finish_id}),
            "avoid_moves": avoid_moves,
            "critic_feedback": self.critic_feedback,
        }))
        if plan.ok:
            path_str = " -> ".join(f"#{p}" for p in plan.path)
            worst = max((m.reach_ratio for m in plan.moves), default=0.0)
            self.log("Planner", "Critic", f"Proposed {len(plan.moves)} moves: {path_str} "
                     f"(hardest move {worst:.0%} of reach)", {"path": plan.path})
        else:
            self.log("Planner", "Orchestrator", plan.error or "No route.")
        return plan

    def critique(self, plan: PlanResponse, rnd: int) -> CritiqueResponse:
        try:
            crit = self._critique(plan, rnd)
        except HTTPException as e:  # Critic down: keep the route, just unreviewed
            crit = CritiqueResponse(approved=False, source="unavailable",
                                    reasoning="The Critic is unavailable, so this route wasn't reviewed.")
            self.log("Critic", "Orchestrator", crit.reasoning, {"error": str(e.detail)})
            return crit
        self.critic_feedback = {"reasoning": crit.reasoning, "avoid_holds": crit.avoid_holds,
                                "avoid_moves": crit.avoid_moves}
        verdict = "APPROVED" if crit.approved else "REJECTED"
        self.log("Critic", "Orchestrator", f"{verdict}: {crit.reasoning}",
                 {"avoid_holds": crit.avoid_holds, "avoid_moves": crit.avoid_moves, "source": crit.source})
        return crit

    def _critique(self, plan: PlanResponse, rnd: int) -> CritiqueResponse:
        return CritiqueResponse(**send(self.client, f"{CRITIC_URL}/critique", {
            "plan": plan.model_dump(),
            "holds": [h.model_dump() for h in self.req.holds],
            "climber": self.req.climber.model_dump(),
            "start_id": self.start_id,
            "finish_id": self.finish_id,
            "round": rnd,
            "max_rounds": self.req.max_rounds,
            "planner_pushback": plan.pushback.model_dump() if plan.pushback else None,
        }))

    def coach(self, plan: PlanResponse) -> CoachResponse:
        try:
            beta = CoachResponse(**send(self.client, f"{COACH_URL}/beta", {
                "plan": plan.model_dump(),
                "holds": [h.model_dump() for h in self.req.holds],
                "climber": self.req.climber.model_dump(),
            }))
        except HTTPException as e:  # Coach down: still return the route, without beta
            beta = CoachResponse(summary="The Coach is unavailable, so there is no move-by-move beta yet.",
                                 steps=[], source="unavailable")
            self.log("Coach", "Orchestrator", beta.summary, {"error": str(e.detail)})
            return beta
        self.log("Coach", "Orchestrator", beta.summary, {"source": beta.source})
        return beta


def same_move(a: list[int], b: list[int]) -> bool:
    return sorted(a) == sorted(b)


# ---------------- fixed pipeline (no LLM) ----------------

def solve_fixed(run: Run) -> tuple[PlanResponse | None, bool, str, int]:
    """Returns (final plan or None, critic approved?, status, rounds)."""
    avoid_holds: set[int] = set()
    avoid_moves: list[list[int]] = []
    last_good: PlanResponse | None = None
    first_plan: PlanResponse | None = None

    for rnd in range(1, run.req.max_rounds + 1):
        constraint_note = (
            f" with constraints: avoid holds {sorted(avoid_holds)}, avoid moves {avoid_moves}"
            if (avoid_holds or avoid_moves) else ""
        )
        run.log("Orchestrator", "Planner", f"Round {rnd}: find a route #{run.start_id} -> #{run.finish_id}{constraint_note}")
        plan = run.plan(avoid_holds, avoid_moves)
        first_plan = first_plan or plan

        if not plan.ok:
            if last_good is None:
                return None, False, plan.error or "No route found.", rnd
            return last_good, False, "The Critic's constraints left no route, so we kept the previous one.", rnd

        last_good = plan
        crit = run.critique(plan, rnd)
        if crit.approved:
            return plan, True, f"Route approved by the Critic in round {rnd}.", rnd

        new_holds = set(crit.avoid_holds) - {run.start_id, run.finish_id} - avoid_holds
        new_moves = [m for m in crit.avoid_moves if not any(same_move(m, old) for old in avoid_moves)]
        if not new_holds and not new_moves:
            return plan, False, "The Critic rejected the route but gave nothing new to change, so we kept it.", rnd
        avoid_holds |= new_holds
        avoid_moves += new_moves

    return last_good, False, f"Hit the {run.req.max_rounds}-round limit; using the last route.", run.req.max_rounds


# ---------------- Claude-run orchestration ----------------

SYSTEM = """You are the orchestrator of a small team of agents that finds a \
bouldering route for one specific climber. You decide what happens next; the \
agents do the work.

- plan_route asks the Planner (pure math, Dijkstra) for the easiest physically \
reachable route. You choose which holds and moves it must avoid. Each call \
starts fresh, so pass the FULL constraint list you want every time.
- critique_route asks the Critic (an experienced coach) to review a plan. It \
approves, or rejects with suggested holds/moves to avoid.
- write_beta asks the Coach to write move-by-move instructions for the plan you \
pick. Calling it finishes the job.

Typical flow: plan, critique, re-plan with the Critic's constraints, repeat \
until approved, then write_beta. Use judgment:
- If a re-plan fails because the constraints left no route, drop the least \
important constraint and try again, or settle for the best earlier plan.
- If the Critic keeps rejecting, or you are out of rounds, pick the best plan \
you have (fewest near-max moves, best holds) and call write_beta.
- Never invent distances or moves; only the Planner's numbers are real.

Give every tool call a short `reason` a climber would understand -- it is \
shown to the user as your part of the conversation. Always finish by calling \
write_beta, unless no plan was ever found."""

REASON = {"type": "string", "description": "One short sentence explaining this step, shown to the user."}
PLAN_ID = {"type": "integer", "description": "The plan_id returned by plan_route."}

TOOLS = [
    {
        "name": "plan_route",
        "description": "Ask the Planner for the easiest reachable route from start to finish, avoiding the given "
                       "holds and moves. Returns a plan_id, the path, and every move's real distance and reach "
                       "ratio -- or an error explaining why no route exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": REASON,
                "avoid_holds": {"type": "array", "items": {"type": "integer"},
                                "description": "Hold ids the route must not use (start/finish are ignored)."},
                "avoid_moves": {"type": "array", "description": "Moves the route must not use, as [from_id, to_id] "
                                "pairs (direction doesn't matter).",
                                "items": {"type": "array", "items": {"type": "integer"}}},
            },
            "required": ["reason", "avoid_holds", "avoid_moves"],
        },
    },
    {
        "name": "critique_route",
        "description": "Ask the Critic to review a plan. Returns approved, its reasoning, and suggested "
                       "avoid_holds / avoid_moves if rejected.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": REASON, "plan_id": PLAN_ID},
            "required": ["reason", "plan_id"],
        },
    },
    {
        "name": "write_beta",
        "description": "Choose the final plan and ask the Coach to write move-by-move beta for it. This ends the job.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": REASON, "plan_id": PLAN_ID},
            "required": ["reason", "plan_id"],
        },
    },
]


def solve_with_claude(run: Run) -> tuple[PlanResponse | None, bool, str, int]:
    """Same return shape as solve_fixed. Raises anthropic.APIError if Claude can't be reached."""
    req = run.req
    llm = anthropic.Anthropic()
    plans: list[PlanResponse] = []  # plan_id = index in this list
    approved_ids: set[int] = set()
    last_error = "No route found."
    plan_calls = critique_calls = 0
    final_id: int | None = None
    final_reason = ""

    def get_plan(tool_input) -> int:
        pid = tool_input.get("plan_id")
        if not isinstance(pid, int) or not 0 <= pid < len(plans) or not plans[pid].ok:
            raise ValueError(f"No successful plan with plan_id {pid}. Valid ids: "
                             f"{[i for i, p in enumerate(plans) if p.ok]}")
        return pid

    def do_tool(name: str, tool_input: dict) -> dict:
        nonlocal plan_calls, critique_calls, last_error, final_id, final_reason
        reason = str(tool_input.get("reason", ""))

        if name == "plan_route":
            if plan_calls >= req.max_rounds:
                raise ValueError(f"Planner limit ({req.max_rounds}) reached. Pick a plan and call write_beta.")
            plan_calls += 1
            avoid_holds = {h for h in tool_input.get("avoid_holds", []) if isinstance(h, int)}
            avoid_moves = [m for m in tool_input.get("avoid_moves", [])
                           if isinstance(m, list) and len(m) == 2 and all(isinstance(x, int) for x in m)]
            note = (f" Avoid holds {sorted(avoid_holds - {run.start_id, run.finish_id})}, avoid moves {avoid_moves}."
                    if (avoid_holds or avoid_moves) else "")
            run.log("Orchestrator", "Planner", f"{reason}{note}", {"source": "claude"})
            plan = run.plan(avoid_holds, avoid_moves)
            plans.append(plan)
            if not plan.ok:
                last_error = plan.error or last_error
                return {"plan_id": len(plans) - 1, "ok": False, "error": plan.error}
            return {"plan_id": len(plans) - 1, "ok": True, "path": plan.path,
                    "moves": [m.model_dump() for m in plan.moves], "planner_calls_left": req.max_rounds - plan_calls}

        if name == "critique_route":
            if critique_calls >= req.max_rounds:
                raise ValueError(f"Critic limit ({req.max_rounds}) reached. Pick a plan and call write_beta.")
            pid = get_plan(tool_input)
            critique_calls += 1
            run.log("Orchestrator", "Critic", f"{reason} (plan {pid})", {"source": "claude"})
            crit = run.critique(plans[pid], critique_calls)
            if crit.approved:
                approved_ids.add(pid)
            return crit.model_dump()

        if name == "write_beta":
            final_id = get_plan(tool_input)
            final_reason = reason
            return {"ok": True}

        raise ValueError(f"Unknown tool {name}")

    messages = [{"role": "user", "content": (
        f"Climber: {req.climber.height_in:.0f} in tall, {req.climber.wingspan_in:.0f} in wingspan, "
        f"max single move {req.climber.reach_in:.0f} in.\n"
        f"Wall: {len(req.holds)} holds. Start #{run.start_id}, finish #{run.finish_id}.\n"
        f"Hold types: {json.dumps({h.id: h.type for h in req.holds})}\n"
        f"You may call plan_route and critique_route up to {req.max_rounds} times each."
    )}]

    # Each round is ~2 tool calls; leave room for retries after errors.
    for _ in range(3 * req.max_rounds + 3):
        response = llm.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages)
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            break  # Claude stopped (or refused / ran out of tokens) -- we pick a plan below

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            try:
                out, is_error = do_tool(tu.name, tu.input), False
            except ValueError as e:
                out, is_error = {"error": str(e)}, True
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(out), "is_error": is_error})
        if final_id is not None:
            break
        messages.append({"role": "user", "content": results})

    good = [i for i, p in enumerate(plans) if p.ok]
    if not good:
        return None, False, last_error, plan_calls

    rounds = max(plan_calls, critique_calls)
    if final_id is None:
        # Claude stopped without choosing: prefer the latest approved plan, else the latest plan.
        final_id = max(approved_ids) if approved_ids else good[-1]
        run.log("Orchestrator", "Coach", f"Claude didn't pick a final plan, so plan {final_id} was used.",
                {"source": "fallback"})
    else:
        run.log("Orchestrator", "Coach", f"{final_reason} (plan {final_id})", {"source": "claude"})

    approved = final_id in approved_ids
    status = (f"Route approved by the Critic after {rounds} round(s)." if approved
              else "The Critic didn't approve any plan; the Orchestrator picked the best one it had.")
    return plans[final_id], approved, status, rounds


# ---------------- the endpoint ----------------

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if len(req.holds) < 2:
        raise HTTPException(400, "Need at least 2 holds.")

    ids = {h.id for h in req.holds}
    # Default start = lowest hold on the wall, finish = highest (image y grows down)
    start_id = req.start_id if req.start_id in ids else max(req.holds, key=lambda h: h.y).id
    finish_id = req.finish_id if req.finish_id in ids else min(req.holds, key=lambda h: h.y).id
    if finish_id == start_id and req.finish_id not in ids:
        # Every hold is at the same height: finish on the one farthest across from the start.
        start = next(h for h in req.holds if h.id == start_id)
        others = [h for h in req.holds if h.id != start_id]
        finish_id = min(others, key=lambda h: (h.y, -abs(h.x - start.x))).id

    with httpx.Client(timeout=TIMEOUT) as client:
        run = Run(client, req, start_id, finish_id)
        source = "fixed pipeline (no API key)"
        if backend() == "claude":
            try:
                plan, approved, status, rounds = solve_with_claude(run)
                source = "claude"
            except anthropic.AnthropicError as e:  # bad key, network, rate limit... keep the demo alive
                source = f"fixed pipeline (Claude call failed: {type(e).__name__})"
                run = Run(client, req, start_id, finish_id)
                run.log("Orchestrator", "Orchestrator", "Claude was unavailable, so the fixed pipeline ran instead.")
        if source != "claude":
            plan, approved, status, rounds = solve_fixed(run)
            if plan is not None:
                run.log("Orchestrator", "Coach", "Write move-by-move beta for the final route.")

        if plan is None:
            return SolveResponse(ok=False, status=status, holds=req.holds, start_id=start_id, finish_id=finish_id,
                                 trace=run.trace, rounds=rounds, orchestrator_source=source)
        beta = run.coach(plan)

    return SolveResponse(
        ok=True, status=status, holds=req.holds, start_id=start_id, finish_id=finish_id,
        plan=plan, critic_approved=approved, beta=beta, trace=run.trace, rounds=rounds,
        orchestrator_source=source,
    )


@app.post("/solve_photo", response_model=SolveResponse)
def solve_photo(req: SolvePhotoRequest):
    """Photo in, route out: Vision -> route-color filter -> /solve (Planner <-> Critic, Coach)."""
    vision_trace: list[TraceEntry] = []

    def log(sender, receiver, summary, payload=None):
        vision_trace.append(TraceEntry(step=len(vision_trace) + 1, sender=sender, receiver=receiver,
                                       summary=summary, payload=payload or {}))

    log("Orchestrator", "Vision", "Detect the holds in this photo.")
    with httpx.Client(timeout=TIMEOUT) as client:
        det = DetectResponse(**send(client, f"{VISION_URL}/detect",
                                    DetectRequest(image_b64=req.image_b64, min_conf=req.min_conf).model_dump()))
    log("Vision", "Orchestrator",
        f"Found {len(det.holds)} holds ({det.method}); colors {det.colors}; "
        f"scale {det.inches_per_pixel} in/px ({det.scale_method})",
        {"colors": det.colors, "inches_per_pixel": det.inches_per_pixel, "scale_method": det.scale_method})

    holds = det.holds
    if len(holds) < 2:
        n = len(holds)
        raise HTTPException(400, f"Vision found {n} hold{'' if n == 1 else 's'} in the photo. "
                                 "Try a straight-on photo of the whole wall in good light.")
    if req.route_color:
        wanted = req.route_color.strip().lower()
        holds = [h for h in holds if (h.color or "").lower() == wanted]
        if len(holds) < 2:
            raise HTTPException(400, f"Only {len(holds)} '{wanted}' holds detected. "
                                     f"Detected colors: {det.colors}")
        # Renumber 0..n-1 (still bottom to top) so start/finish ids refer to this route's holds.
        holds = [Hold(id=i, x=h.x, y=h.y, type=h.type, color=h.color) for i, h in enumerate(holds)]
        log("Orchestrator", "Orchestrator", f"Kept {len(holds)} {wanted} holds for the route.")

    ipp = req.inches_per_pixel or det.inches_per_pixel
    if not ipp:
        raise HTTPException(400, "No scale available: pass inches_per_pixel.")

    result = solve(SolveRequest(
        holds=holds,
        start_id=req.start_id,
        finish_id=req.finish_id,
        inches_per_pixel=ipp,
        climber=req.climber,
        max_rounds=req.max_rounds,
    ))

    # One continuous trace: Vision messages first, then the Planner/Critic/Coach rounds.
    offset = len(vision_trace)
    for entry in result.trace:
        entry.step += offset
    result.trace = vision_trace + result.trace
    return result
