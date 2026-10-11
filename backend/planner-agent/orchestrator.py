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
import queue
import sys
import threading
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

load_dotenv()  # reads ANTHROPIC_API_KEY (and the *_URL overrides) from .env

# Make the repo root importable so `shared` works however this file is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.shared.llm import MODEL, backend, llm_available  # noqa: E402
from backend.shared.schemas import (  # noqa: E402
    CoachResponse,
    CritiqueResponse,
    DetectRequest,
    DetectResponse,
    HOLD_TYPES,
    GripPick,
    Hold,
    PlanResponse,
    RouteDifficulty,
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


# ---------------- plain-language descriptions (for the trace the user reads) ----------------

def where(holds: list[Hold], hid: int) -> str:
    """'the crimp high on the left': a hold described the way a climber would point at it."""
    h = next((h for h in holds if h.id == hid), None)
    if h is None:
        return "that hold"
    xs, ys = [o.x for o in holds], [o.y for o in holds]
    fx = (h.x - min(xs)) / ((max(xs) - min(xs)) or 1)
    fy = (h.y - min(ys)) / ((max(ys) - min(ys)) or 1)  # image y grows down
    height = "high up" if fy < 1 / 3 else "down low" if fy > 2 / 3 else "in the middle"
    side = "on the left" if fx < 1 / 3 else "on the right" if fx > 2 / 3 else ""
    kind = h.type if h.type in HOLD_TYPES else "hold"
    if height == "in the middle" and not side:
        return f"the {kind} in the middle"
    return f"the {kind} {height}" + (f" {side}" if side else "")


def and_list(items: list[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def avoid_note(holds: list[Hold], path: list[int], avoid_holds, avoid_moves) -> str:
    """'skip the sloper high on the left and move 3' -- moves numbered like the route on screen."""
    parts = [where(holds, h) for h in sorted(avoid_holds)]
    for a, b in avoid_moves:
        k = next((i + 1 for i in range(len(path) - 1) if {path[i], path[i + 1]} == {a, b}), None)
        far = b if k is None or path[k] == b else a  # the hold that move lands on
        parts.append(f"move {k}, the reach to {where(holds, far)}" if k
                     else f"the move from {where(holds, a)} to {where(holds, b)}")
    return and_list(parts) if parts else ""


def drawing(holds: list[Hold], path=(), highlights=(), marks=(), crossed=()) -> dict:
    """What the page should sketch on the photo for one message, in image pixels.

    path: hold ids to connect in order; highlights: (from_id, to_id, label) moves to call out;
    marks: (hold_id, label) holds to label; crossed: hold ids to strike out."""
    xy = {h.id: [h.x, h.y] for h in holds}
    return {
        "path": [xy[i] for i in path if i in xy],
        "highlights": [{"from": xy[a], "to": xy[b], "label": label}
                       for a, b, label in highlights if a in xy and b in xy],
        "marks": [{"at": xy[i], "label": label} for i, label in marks if i in xy],
        "crossed": [xy[i] for i in crossed if i in xy],
    }


def skip_drawing(holds: list[Hold], path: list[int], start_id: int, finish_id: int,
                 avoid_holds, avoid_moves) -> dict:
    """Start and finish labelled, plus the holds and moves the Planner is told to skip."""
    return drawing(holds, marks=[(start_id, "start"), (finish_id, "finish")], crossed=sorted(avoid_holds),
                   highlights=[(a, b, "skip") for a, b in avoid_moves])


def nearest_hold(holds: list[Hold], point: tuple[float, float]) -> int:
    return min(holds, key=lambda h: (h.x - point[0]) ** 2 + (h.y - point[1]) ** 2).id


GRIP_SNAP_PX = 30  # a grip only counts if it was set on (about) this hold, not one of another color


def grips_to_hold_types(holds: list[Hold], grips: list[GripPick]) -> dict[int, str]:
    """Turn grips tapped on the photo into hold_types for this route's hold ids."""
    by_id = {h.id: h for h in holds}
    out = {}
    for g in grips:
        hid = nearest_hold(holds, (g.x, g.y))
        h = by_id[hid]
        if (h.x - g.x) ** 2 + (h.y - g.y) ** 2 <= GRIP_SNAP_PX ** 2:
            out[hid] = g.type
    return out


# ---------------- live trace (for /solve_photo/stream) ----------------

_live = threading.local()  # .listener: called with each TraceEntry as it is logged, on this thread


def emit(entry: TraceEntry):
    listener = getattr(_live, "listener", None)
    if listener:
        listener(entry)


# ---------------- calls to the other agents (shared by both modes) ----------------

class Run:
    """One /solve request: the agents' HTTP client plus the trace being recorded."""

    def __init__(self, client: httpx.Client, req: SolveRequest, start_id: int, finish_id: int):
        self.client, self.req = client, req
        self.start_id, self.finish_id = start_id, finish_id
        self.trace: list[TraceEntry] = []
        self.critic_feedback: dict | None = None  # last verdict, so the Planner can push back on it
        self.last_path: list[int] = []  # last proposed route, so avoided moves can be named by number

    def log(self, sender, receiver, summary, payload=None):
        self.trace.append(TraceEntry(step=len(self.trace) + 1, sender=sender, receiver=receiver,
                                     summary=summary, payload=payload or {}))
        emit(self.trace[-1])

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
            self.last_path = plan.path
            n = len(plan.moves)
            summary = f"Here's a route with {n} move{'' if n == 1 else 's'}."
            if plan.moves:
                k, hard = max(enumerate(plan.moves, 1), key=lambda km: km[1].reach_ratio)
                kind = hard.to_type if hard.to_type in HOLD_TYPES else "hold"
                summary += (f" The hardest is move {k}, about {hard.distance_in:.0f} in to a {kind}, "
                            f"{hard.reach_ratio:.0%} of your reach.")
            hardest = [(hard.from_id, hard.to_id, f"move {k} · {hard.distance_in:.0f} in")] if plan.moves else []
            self.log("Planner", "Critic", summary + " Does it look OK?", {
                "path": plan.path,
                "draw": drawing(self.req.holds, path=plan.path, highlights=hardest,
                                marks=[(self.start_id, "start"), (self.finish_id, "finish")]),
            })
        else:
            self.log("Planner", "Orchestrator", plan.error or "I couldn't find a route.")
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
        verdict = "Looks good." if crit.approved else "Not quite."
        self.log("Critic", "Orchestrator", f"{verdict} {crit.reasoning}".strip(),
                 {"approved": crit.approved, "avoid_holds": crit.avoid_holds, "avoid_moves": crit.avoid_moves,
                  "source": crit.source,
                  "draw": drawing(self.req.holds, path=plan.path, crossed=crit.avoid_holds,
                                  highlights=[(a, b, "too far") for a, b in crit.avoid_moves])})
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

    def log_hold_types(self, assumed_holds: list[int]):
        req = self.req
        notes = []
        chosen = req.hold_types
        if 0 < len(chosen) <= 2:
            notes.append("You marked " + and_list([
                f"{where(req.holds, i).replace(' ' + t + ' ', ' hold ')} as a {t}"
                for i, t in sorted(chosen.items())]) + ".")
        elif chosen:
            counts = {t: list(chosen.values()).count(t) for t in dict.fromkeys(sorted(chosen.values()))}
            notes.append(f"You set grips on {len(chosen)} holds: "
                         + and_list([f"{n} {t}{'' if n == 1 else 's'}" for t, n in counts.items()]) + ".")
        if assumed_holds:
            n = len(assumed_holds)
            notes.append(f"I'll treat the other {n} as {req.default_hold_type}s." if chosen and n > 1 else
                         f"I couldn't tell what kind of hold {n} of these {'is' if n == 1 else 'are'}, "
                         f"so I'll treat {'it as a' if n == 1 else 'them as'} {req.default_hold_type}"
                         f"{'' if n == 1 else 's'}.")
        if not notes:
            return
        self.log("Orchestrator", "Orchestrator", " ".join(notes),
                 {"hold_types": req.hold_types, "assumed_holds": assumed_holds,
                  "default_hold_type": req.default_hold_type,
                  "draw": drawing(req.holds, marks=sorted(req.hold_types.items()))})

    def rate(self, plan: PlanResponse, assumed_holds: list[int]) -> RouteDifficulty | None:
        """Ask the Critic how hard the final route is, from its hold types and reach."""
        try:
            diff = RouteDifficulty(**send(self.client, f"{CRITIC_URL}/difficulty", {
                "plan": plan.model_dump(),
                "holds": [h.model_dump() for h in self.req.holds],
                "climber": self.req.climber.model_dump(),
                "assumed_holds": assumed_holds,
                "default_hold_type": self.req.default_hold_type,
            }))
        except HTTPException as e:  # Critic down: the route is still usable, just ungraded
            self.log("Critic", "Orchestrator", "The Critic is unavailable, so the route wasn't graded.",
                     {"error": str(e.detail)})
            return None
        self.log("Critic", "Orchestrator", f"I'd call this a {diff.grade}, {diff.label}. {diff.reasoning}",
                 {"grade": diff.grade, "score": diff.score, "hold_counts": diff.hold_counts,
                  "assumed_holds": diff.assumed_holds, "source": diff.source,
                  "draw": drawing(self.req.holds, path=plan.path, highlights=[
                      (diff.hardest_move.from_id, diff.hardest_move.to_id, "crux")] if diff.hardest_move else [])})
        return diff

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


def apply_hold_types(req: SolveRequest) -> tuple[SolveRequest, list[int]]:
    """Fill in hold types: the user's choice wins, then Vision's, then the default.

    Returns the updated request and the ids whose type was assumed (the default)."""
    ids = {h.id for h in req.holds}
    unknown_ids = sorted(set(req.hold_types) - ids)
    if unknown_ids:
        raise HTTPException(400, f"hold_types names holds {unknown_ids} that aren't on this route. "
                                 f"Valid ids: 0-{max(ids)}.")
    holds, assumed = [], []
    for h in req.holds:
        if h.id in req.hold_types:
            t = req.hold_types[h.id]
        elif h.type.lower() in HOLD_TYPES:
            t = h.type.lower()
        else:
            t = req.default_hold_type
            assumed.append(h.id)
        holds.append(h.model_copy(update={"type": t}))
    return req.model_copy(update={"holds": holds}), assumed


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
        holds = run.req.holds
        skip = avoid_note(holds, run.last_path, avoid_holds, avoid_moves)
        run.log("Orchestrator", "Planner",
                f"Let's try again, but skip {skip}." if skip else
                f"Find a way from {where(holds, run.start_id)} up to {where(holds, run.finish_id)}.",
                {"round": rnd, "draw": skip_drawing(holds, run.last_path, run.start_id, run.finish_id,
                                                    avoid_holds, avoid_moves)})
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
shown to the user as your part of the conversation. Write it like a friendly \
teammate talking to the agent you're calling ("Critic, can you check this one? \
Move 3 looks long."). Refer to moves by number (move 1 is the first move) and to \
holds by type and position ("the crimp high on the left"), never by hold id, \
and round distances to whole inches. Always finish by calling \
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
            skip = avoid_note(req.holds, run.last_path, avoid_holds - {run.start_id, run.finish_id}, avoid_moves)
            note = f" Skip {skip}." if skip else ""
            run.log("Orchestrator", "Planner", f"{reason}{note}", {
                "source": "claude",
                "draw": skip_drawing(req.holds, run.last_path, run.start_id, run.finish_id,
                                     avoid_holds - {run.start_id, run.finish_id}, avoid_moves)})
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
            run.log("Orchestrator", "Critic", reason, {"source": "claude", "plan_id": pid})
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
        run.log("Orchestrator", "Coach", "Let's go with the best route we found. Coach, can you write up the beta?",
                {"source": "fallback", "plan_id": final_id})
    else:
        run.log("Orchestrator", "Coach", final_reason, {"source": "claude", "plan_id": final_id})

    approved = final_id in approved_ids
    status = (f"Route approved by the Critic after {rounds} round(s)." if approved
              else "The Critic didn't approve any plan; the Orchestrator picked the best one it had.")
    return plans[final_id], approved, status, rounds


# ---------------- the endpoint ----------------

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if len(req.holds) < 2:
        raise HTTPException(400, "Need at least 2 holds.")
    req, assumed_holds = apply_hold_types(req)

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
        run.log_hold_types(assumed_holds)
        source = "fixed pipeline (no API key)"
        if backend() == "claude":
            try:
                plan, approved, status, rounds = solve_with_claude(run)
                source = "claude"
            except anthropic.AnthropicError as e:  # bad key, network, rate limit... keep the demo alive
                source = f"fixed pipeline (Claude call failed: {type(e).__name__})"
                run = Run(client, req, start_id, finish_id)
                run.log_hold_types(assumed_holds)
                run.log("Orchestrator", "Orchestrator", "Claude was unavailable, so the fixed pipeline ran instead.")
        if source != "claude":
            plan, approved, status, rounds = solve_fixed(run)
            if plan is not None:
                run.log("Orchestrator", "Coach", "Coach, can you walk the climber through this route move by move?")

        if plan is None:
            return SolveResponse(ok=False, status=status, holds=req.holds, start_id=start_id, finish_id=finish_id,
                                 trace=run.trace, rounds=rounds, orchestrator_source=source)
        difficulty = run.rate(plan, assumed_holds)
        beta = run.coach(plan)

    return SolveResponse(
        ok=True, status=status, holds=req.holds, start_id=start_id, finish_id=finish_id,
        plan=plan, critic_approved=approved, difficulty=difficulty, beta=beta, trace=run.trace, rounds=rounds,
        orchestrator_source=source,
    )


@app.post("/solve_photo", response_model=SolveResponse)
def solve_photo(req: SolvePhotoRequest):
    """Photo in, route out: Vision -> route-color filter -> /solve (Planner <-> Critic, Coach)."""
    vision_trace: list[TraceEntry] = []

    def log(sender, receiver, summary, payload=None):
        vision_trace.append(TraceEntry(step=len(vision_trace) + 1, sender=sender, receiver=receiver,
                                       summary=summary, payload=payload or {}))
        emit(vision_trace[-1])

    log("Orchestrator", "Vision", "Vision, what holds can you see in this photo?")
    with httpx.Client(timeout=TIMEOUT) as client:
        det = DetectResponse(**send(client, f"{VISION_URL}/detect",
                                    DetectRequest(image_b64=req.image_b64, min_conf=req.min_conf).model_dump()))
    by_color = [f"{n} {c}" for c, n in sorted(det.colors.items(), key=lambda cn: -cn[1])]
    log("Vision", "Orchestrator",
        f"I found {len(det.holds)} hold{'' if len(det.holds) == 1 else 's'}"
        + (f": {and_list(by_color)}." if by_color else "."),
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
        log("Orchestrator", "Orchestrator", f"We'll stick to the {len(holds)} {wanted} holds.")

    # A start/finish tapped on the photo becomes the nearest hold on this route.
    start_id, finish_id = req.start_id, req.finish_id
    if req.start_point:
        start_id = nearest_hold(holds, req.start_point)
        log("Orchestrator", "Orchestrator", f"You picked {where(holds, start_id)} as the start.",
            {"draw": drawing(holds, marks=[(start_id, "start")])})
    if req.finish_point:
        finish_id = nearest_hold(holds, req.finish_point)
        if finish_id == start_id:
            raise HTTPException(400, "The start and finish are the same hold. Pick two different holds.")
        log("Orchestrator", "Orchestrator", f"You picked {where(holds, finish_id)} as the finish.",
            {"draw": drawing(holds, marks=[(finish_id, "finish")])})

    if req.scale_ref:
        ipp = req.scale_ref.inches_per_pixel
        log("Orchestrator", "Orchestrator", "I'll size every move from your measurement of the wall, "
            "not from the hold sizes.", {"inches_per_pixel": ipp, "scale_method": "user-measurement"})
    else:
        ipp = req.inches_per_pixel or det.inches_per_pixel
    if not ipp:
        raise HTTPException(400, "No scale available: pass inches_per_pixel.")

    result = solve(SolveRequest(
        holds=holds,
        start_id=start_id,
        finish_id=finish_id,
        inches_per_pixel=ipp,
        climber=req.climber,
        max_rounds=req.max_rounds,
        hold_types={**req.hold_types, **grips_to_hold_types(holds, req.grips)},
        default_hold_type=req.default_hold_type,
    ))

    # One continuous trace: Vision messages first, then the Planner/Critic/Coach rounds.
    offset = len(vision_trace)
    for entry in result.trace:
        entry.step += offset
    result.trace = vision_trace + result.trace
    return result


@app.post("/solve_photo/stream")
def solve_photo_stream(req: SolvePhotoRequest):
    """Same as /solve_photo, but streams the agents' messages as they happen.

    Newline-delimited JSON: {"type": "trace", "entry": ...} for each message, then one
    {"type": "result", "result": SolveResponse} or {"type": "error", "detail": str}.
    """
    events: queue.Queue = queue.Queue()
    step = 0

    def on_entry(entry: TraceEntry):
        nonlocal step
        step += 1  # live numbering; the final result carries the canonical trace
        events.put({"type": "trace", "entry": {**entry.model_dump(), "step": step}})

    def work():
        _live.listener = on_entry
        try:
            events.put({"type": "result", "result": solve_photo(req).model_dump()})
        except HTTPException as e:
            events.put({"type": "error", "detail": str(e.detail)})
        except Exception as e:  # keep the stream well-formed so the page can show the error
            events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            _live.listener = None
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
