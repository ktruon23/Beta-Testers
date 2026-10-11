"""Planner Agent: Claude chooses planning strategy; Python owns all geometry.

The route-search helpers are deterministic. When Claude is available, it chooses
which helpers to call and submits a justified route. If Claude is unavailable or
fails, the service returns a balanced deterministic route.
"""

from __future__ import annotations

import heapq
import json
import math
from typing import Any

from fastapi import FastAPI

from shared.llm import call_structured, llm_available
from shared.schemas import (
    Hold,
    Move,
    PlanAlternative,
    PlanRequest,
    PlanResponse,
    PlannerPushback,
)

app = FastAPI(title="Planner Agent")

# Lower penalty means the destination hold is generally more comfortable.
HOLD_PENALTY = {
    "jug": 0.0,
    "pinch": 0.4,
    "pocket": 0.6,
    "crimp": 0.8,
    "sloper": 1.0,
    "unknown": 0.5,
}

PRESETS: dict[str, dict[str, float]] = {
    "safe": {"move": 1.0, "reach": 8.0, "hold_type": 1.5},
    "balanced": {"move": 1.0, "reach": 4.0, "hold_type": 1.0},
    "efficient": {"move": 3.0, "reach": 2.0, "hold_type": 0.5},
}

# How the climber's level changes the search. Intermediate is the neutral baseline.
# reach_cap: fraction of full reach a single move may use (hard limit).
# weight_scale: multiplies the strategy weights for move count, reach and hold type.
# extra_hold_penalty: added on top of HOLD_PENALTY for holds that are hard for this level.
LEVEL_PROFILES: dict[str, dict[str, Any]] = {
    "beginner": {
        "reach_cap": 0.8,
        "weight_scale": {"move": 0.75, "reach": 2.0, "hold_type": 2.0},
        "extra_hold_penalty": {"crimp": 0.6, "sloper": 0.6, "pocket": 0.4},
        "default_strategy": "safe",
    },
    "intermediate": {
        "reach_cap": 1.0,
        "weight_scale": {"move": 1.0, "reach": 1.0, "hold_type": 1.0},
        "extra_hold_penalty": {},
        "default_strategy": "balanced",
    },
    "expert": {
        "reach_cap": 1.0,
        "weight_scale": {"move": 1.5, "reach": 0.5, "hold_type": 0.5},
        "extra_hold_penalty": {},
        "default_strategy": "efficient",
    },
}


def level_profile(level: str | None) -> dict[str, Any]:
    return LEVEL_PROFILES.get((level or "intermediate").lower(), LEVEL_PROFILES["intermediate"])


SYSTEM = """You are the decision-making layer of a climbing route Planner Agent.
The Python tools are authoritative for all geometry, distances, reach ratios,
and route paths. Never calculate or invent geometry yourself.

Choose a strategy appropriate to the climber: beginners generally need shorter,
safer moves; expert climbers may prefer fewer moves. Compare at least two
candidate routes when possible. If no route exists, diagnose the gap with
find_gaps. Then submit a path that exactly matches one of the tool results, with
brief reasoning. If critic feedback makes the constrained route much worse, use
numeric evidence to propose a specific alternative constraint. Never suggest
exceeding the climber's computed reach. Use no more than five tool calls.
"""

ACTION_TOOL = {
    "name": "planner_action",
    "description": "Choose the next deterministic planning tool to run, or submit the final plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["plan_route", "find_gaps", "check_move", "submit_plan"],
            },
            "strategy": {"type": "string", "enum": ["safe", "balanced", "efficient", "custom"]},
            "weights": {
                "type": "object",
                "properties": {
                    "move": {"type": "number"},
                    "reach": {"type": "number"},
                    "hold_type": {"type": "number"},
                },
                "required": ["move", "reach", "hold_type"],
            },
            "from_id": {"type": "integer"},
            "to_id": {"type": "integer"},
            "path": {"type": "array", "items": {"type": "integer"}},
            "chosen_strategy": {"type": "string"},
            "reasoning": {"type": "string"},
            "pushback": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "proposal": {"type": "string"},
                },
                "required": ["reason", "proposal"],
            },
        },
        "required": ["action"],
    },
}


def make_move(a: Hold, b: Hold, ipp: float, reach_in: float) -> Move:
    """Compute a move from pixel coordinates; y is flipped into upward-positive inches."""
    dx = (b.x - a.x) * ipp
    dy = (a.y - b.y) * ipp
    distance = math.hypot(dx, dy)
    return Move(
        from_id=a.id,
        to_id=b.id,
        distance_in=round(distance, 2),
        reach_ratio=round(distance / reach_in, 4),
        dx_in=round(dx, 2),
        dy_in=round(dy, 2),
        from_type=a.type,
        to_type=b.type,
    )


def _cost(move: Move, weights: dict[str, float], profile: dict[str, Any] | None = None) -> float:
    profile = profile or LEVEL_PROFILES["intermediate"]
    scale = profile["weight_scale"]
    hold = move.to_type.lower()
    penalty = HOLD_PENALTY.get(hold, HOLD_PENALTY["unknown"]) + profile["extra_hold_penalty"].get(hold, 0.0)
    return (
        weights["move"] * scale["move"]
        + weights["reach"] * scale["reach"] * (move.reach_ratio**2)
        + weights["hold_type"] * scale["hold_type"] * penalty
    )


def plan_route(
    req: PlanRequest,
    weights: dict[str, float] | None = None,
    avoid_holds: list[int] | None = None,
    avoid_moves: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Deterministic Dijkstra tool. Returns a path plus exact move measurements.

    The climber's level scales the cost weights and, for beginners, caps how much of
    their reach a single move may use. If the cap leaves no route, it is relaxed to
    full reach and the result says so (level_cap_relaxed).
    """
    profile = level_profile(getattr(req.climber, "level", None))
    result = _search(req, weights, avoid_holds, avoid_moves, profile, profile["reach_cap"])
    if not result["ok"] and profile["reach_cap"] < 1.0 and "closest_reachable_id" in result:
        relaxed = _search(req, weights, avoid_holds, avoid_moves, profile, 1.0)
        if relaxed["ok"]:
            relaxed["level_cap_relaxed"] = True
            return relaxed
    result.setdefault("level_cap_relaxed", False)
    return result


def _search(
    req: PlanRequest,
    weights: dict[str, float] | None,
    avoid_holds: list[int] | None,
    avoid_moves: list[list[int]] | None,
    profile: dict[str, Any],
    reach_cap: float,
) -> dict[str, Any]:
    weights = weights or PRESETS["balanced"]
    avoided_holds = set(req.avoid_holds if avoid_holds is None else avoid_holds)
    avoided_moves = {
        frozenset(pair) for pair in (req.avoid_moves if avoid_moves is None else avoid_moves)
        if len(pair) == 2
    }
    by_id = {h.id: h for h in req.holds}
    reach_in = req.climber.reach_in
    move_limit = reach_in * reach_cap

    if req.start_id not in by_id or req.finish_id not in by_id:
        return {"ok": False, "error": "Start or finish hold ID was not found.", "path": [], "moves": [], "total_cost": 0.0, "reach_in": reach_in}
    if req.start_id == req.finish_id:
        return {"ok": True, "path": [req.start_id], "moves": [], "total_cost": 0.0, "reach_in": reach_in, "max_reach_ratio": 0.0}

    active = [h for h in req.holds if h.id not in avoided_holds or h.id in (req.start_id, req.finish_id)]
    start, finish = by_id[req.start_id], by_id[req.finish_id]
    if not any(h.id == start.id for h in active) or not any(h.id == finish.id for h in active):
        return {"ok": False, "error": "The start or finish hold is excluded.", "path": [], "moves": [], "total_cost": 0.0, "reach_in": reach_in}

    # Distances are computed before rounding; reachability is a hard constraint.
    moves_by_from: dict[int, list[tuple[int, Move, float]]] = {h.id: [] for h in active}
    for a in active:
        for b in active:
            if a.id == b.id or frozenset((a.id, b.id)) in avoided_moves:
                continue
            move = make_move(a, b, req.inches_per_pixel, reach_in)
            exact_distance = math.hypot((b.x - a.x) * req.inches_per_pixel, (a.y - b.y) * req.inches_per_pixel)
            if exact_distance <= move_limit + 1e-9:
                moves_by_from[a.id].append((b.id, move, _cost(move, weights, profile)))

    best: dict[int, float] = {req.start_id: 0.0}
    came_from: dict[int, tuple[int, Move]] = {}
    heap: list[tuple[float, int]] = [(0.0, req.start_id)]
    done: set[int] = set()
    while heap:
        cost_so_far, current = heapq.heappop(heap)
        if current in done:
            continue
        done.add(current)
        if current == req.finish_id:
            break
        for nxt, move, edge_cost in moves_by_from.get(current, []):
            new_cost = cost_so_far + edge_cost
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                came_from[nxt] = (current, move)
                heapq.heappush(heap, (new_cost, nxt))

    if req.finish_id not in best:
        reachable = done
        closest = min(reachable, key=lambda hid: math.hypot(
            (by_id[hid].x - finish.x) * req.inches_per_pixel,
            (by_id[hid].y - finish.y) * req.inches_per_pixel,
        )) if reachable else req.start_id
        gap_distance = math.hypot(
            (by_id[closest].x - finish.x) * req.inches_per_pixel,
            (by_id[closest].y - finish.y) * req.inches_per_pixel,
        )
        return {
            "ok": False,
            "error": f"No route reaches finish hold #{req.finish_id}. Closest reachable hold is #{closest}, {gap_distance:.1f} in from the finish (max move {move_limit:.1f} in).",
            "path": [], "moves": [], "total_cost": 0.0, "reach_in": reach_in,
            "closest_reachable_id": closest, "gap_distance_in": round(gap_distance, 2),
        }

    path = [req.finish_id]
    moves: list[Move] = []
    cursor = req.finish_id
    while cursor != req.start_id:
        previous, move = came_from[cursor]
        moves.append(move)
        path.append(previous)
        cursor = previous
    path.reverse()
    moves.reverse()
    return {
        "ok": True,
        "path": path,
        "moves": moves,
        "total_cost": round(best[req.finish_id], 4),
        "reach_in": reach_in,
        "max_reach_ratio": max((m.reach_ratio for m in moves), default=0.0),
        "move_count": len(moves),
        "worst_hold_type": max(moves, key=lambda m: m.reach_ratio).to_type if moves else "none",
        "strategy_weights": weights,
        "move_limit_in": round(move_limit, 2),
    }


def check_move(req: PlanRequest, from_id: int, to_id: int) -> dict[str, Any]:
    by_id = {h.id: h for h in req.holds}
    if from_id not in by_id or to_id not in by_id:
        return {"ok": False, "error": f"Unknown hold ID in move #{from_id} -> #{to_id}."}
    move = make_move(by_id[from_id], by_id[to_id], req.inches_per_pixel, req.climber.reach_in)
    return {"ok": True, **move.model_dump(), "reachable": move.distance_in <= req.climber.reach_in + 1e-9}


def find_gaps(req: PlanRequest) -> dict[str, Any]:
    """Find holds reachable from start and report closest frontier-to-finish gaps."""
    base = plan_route(req, PRESETS["balanced"], avoid_holds=[], avoid_moves=[])
    if base["ok"]:
        return {"route_exists": True, "reachable_from_start": base["path"], "gaps": []}

    by_id = {h.id: h for h in req.holds}
    start = by_id.get(req.start_id)
    finish = by_id.get(req.finish_id)
    if start is None or finish is None:
        return {"route_exists": False, "gaps": [], "diagnosis": base.get("error", "Invalid endpoints.")}
    reach = req.climber.reach_in
    reachable: set[int] = set()
    stack = [req.start_id]
    while stack:
        cur = stack.pop()
        if cur in reachable or cur not in by_id:
            continue
        reachable.add(cur)
        a = by_id[cur]
        for b in req.holds:
            if b.id in reachable:
                continue
            distance = math.hypot((b.x-a.x)*req.inches_per_pixel, (a.y-b.y)*req.inches_per_pixel)
            if distance <= reach + 1e-9:
                stack.append(b.id)
    frontier = [by_id[i] for i in reachable]
    gaps = []
    for a in frontier:
        distance = math.hypot((finish.x-a.x)*req.inches_per_pixel, (a.y-finish.y)*req.inches_per_pixel)
        if distance > reach:
            gaps.append({"from_id": a.id, "to_id": finish.id, "distance_in": round(distance, 2), "reach_in": round(reach, 2), "over_by_in": round(distance-reach, 2)})
    gaps.sort(key=lambda g: g["over_by_in"])
    return {
        "route_exists": False,
        "reachable_from_start": sorted(reachable),
        "gaps": gaps[:3],
        "diagnosis": base.get("error", "No route found under the current reach limit."),
    }


def _candidate_summary(result: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "ok": result.get("ok", False),
        "path": result.get("path", []),
        "total_cost": result.get("total_cost", 0.0),
        "move_count": result.get("move_count", len(result.get("moves", []))),
        "max_reach_ratio": result.get("max_reach_ratio", max((m.reach_ratio for m in result.get("moves", [])), default=0.0)),
        "worst_hold_type": result.get("worst_hold_type", "unknown"),
        "error": result.get("error"),
    }


def _deterministic_response(req: PlanRequest, reason: str | None = None) -> PlanResponse:
    level = getattr(req.climber, "level", None) or "intermediate"
    default = level_profile(level)["default_strategy"]
    reason = reason or f"Claude unavailable; {default} deterministic planner used ({level} climber)."
    balanced = plan_route(req, PRESETS[default])
    if not balanced["ok"]:
        gaps = find_gaps(req)
        diagnostic = gaps.get("diagnosis") or balanced.get("error") or "No reachable route was found."
        return PlanResponse(
            ok=False, path=[], moves=[], reach_in=req.climber.reach_in,
            total_cost=0.0, error=diagnostic, strategy=default,
            reasoning=f"{reason} {diagnostic}", alternatives=[], pushback=None, source="deterministic",
        )
    if balanced.get("level_cap_relaxed"):
        reason += " No route fit the beginner reach limit, so full reach was allowed."
    alternatives = []
    for name in (n for n in PRESETS if n != default):
        candidate = plan_route(req, PRESETS[name])
        if candidate["ok"] and candidate["path"] != balanced["path"]:
            alternatives.append(PlanAlternative(**_candidate_summary(candidate, name)))
    return PlanResponse(
        ok=True, path=balanced["path"], moves=balanced["moves"], reach_in=req.climber.reach_in,
        total_cost=balanced["total_cost"], error=None, strategy=default, reasoning=reason,
        alternatives=alternatives, pushback=None, source="deterministic",
    )


def _build_pushback(req: PlanRequest, chosen: dict[str, Any], strategy: str) -> PlannerPushback | None:
    if not (req.avoid_holds or req.avoid_moves) or not chosen.get("ok"):
        return None
    original = plan_route(req, PRESETS["balanced"], avoid_holds=[], avoid_moves=[])
    if not original.get("ok"):
        return None
    original_max = original.get("max_reach_ratio", 0.0)
    constrained_max = chosen.get("max_reach_ratio", 0.0)
    cost_ratio = chosen.get("total_cost", 0.0) / max(original.get("total_cost", 0.0), 1e-9)
    if constrained_max - original_max < 0.12 and cost_ratio < 1.4:
        return None
    avoided = sorted(set(req.avoid_holds))
    reason = (
        f"The constraints change the maximum reach ratio from {original_max:.0%} to "
        f"{constrained_max:.0%} and route cost from {original['total_cost']:.2f} to {chosen['total_cost']:.2f}."
    )
    proposal = "Keep the safer original route and avoid a specific risky move instead of excluding an entire hold." if avoided else "Reconsider the avoided move; the constrained route is substantially more demanding."
    return PlannerPushback(
        constraint={"avoid_holds": avoided, "avoid_moves": req.avoid_moves},
        reason=reason,
        evidence={"original_max_reach": float(original_max), "constrained_max_reach": float(constrained_max), "original_cost": float(original["total_cost"]), "constrained_cost": float(chosen["total_cost"])},
        proposal=proposal,
    )


def _llm_plan(req: PlanRequest) -> PlanResponse:
    # Results are computed by Python; Claude only selects actions/strategies and explains them.
    tool_results: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    gaps_result: dict[str, Any] | None = None
    final_action: dict[str, Any] | None = None
    last_error: str | None = None

    level = getattr(req.climber, "level", None) or "intermediate"
    feedback = getattr(req, "critic_feedback", None)
    user_context = {
        "request": req.model_dump(),
        "climber_level": level,
        "level_profile": level_profile(level),
        "critic_feedback": feedback.model_dump() if hasattr(feedback, "model_dump") else feedback,
        "available_presets": PRESETS,
        "tool_results_so_far": tool_results,
        "instructions": "Choose one action now. Run at least two distinct plan_route presets when possible, then submit_plan. Each response can request exactly one action.",
    }

    for call_index in range(5):
        user_context["tool_results_so_far"] = tool_results
        user_context["candidate_summaries"] = [_candidate_summary(v, k) for k, v in candidates.items()]
        try:
            action = call_structured(SYSTEM, json.dumps(user_context, default=str), ACTION_TOOL)
        except Exception as exc:
            last_error = f"Claude call failed: {type(exc).__name__}"
            break
        if not isinstance(action, dict):
            last_error = "Claude returned an invalid action."
            break

        name = action.get("action")
        if name == "plan_route":
            strategy = str(action.get("strategy", "balanced"))
            if strategy not in PRESETS:
                strategy = "custom"
            weights = action.get("weights") if strategy == "custom" else PRESETS[strategy]
            if not isinstance(weights, dict) or not all(k in weights for k in ("move", "reach", "hold_type")):
                weights = PRESETS["balanced"]
                strategy = "balanced"
            weights = {k: max(0.0, min(float(weights[k]), 20.0)) for k in ("move", "reach", "hold_type")}
            result = plan_route(req, weights)
            candidates[strategy] = result
            tool_results.append({"tool": "plan_route", "strategy": strategy, "result": _candidate_summary(result, strategy)})
        elif name == "find_gaps":
            gaps_result = find_gaps(req)
            tool_results.append({"tool": "find_gaps", "result": gaps_result})
        elif name == "check_move":
            result = check_move(req, int(action.get("from_id", req.start_id)), int(action.get("to_id", req.finish_id)))
            tool_results.append({"tool": "check_move", "result": result})
        elif name == "submit_plan":
            if len(candidates) < 2 and call_index < 4:
                tool_results.append({"tool": "submit_plan", "result": "Please compare at least two distinct strategy candidates before submitting, if routes are available."})
                continue
            final_action = action
            tool_results.append({"tool": "submit_plan", "result": {"chosen_strategy": action.get("chosen_strategy"), "reasoning": action.get("reasoning")}})
            break
        else:
            tool_results.append({"tool": "invalid_action", "result": "Unknown action. Choose a documented action."})

    if last_error:
        return _deterministic_response(req, last_error + "; deterministic fallback used.")

    valid_candidates = {k: v for k, v in candidates.items() if v.get("ok")}
    if not valid_candidates:
        diagnostic = (gaps_result or find_gaps(req)).get("diagnosis", "No reachable route was found.")
        return PlanResponse(
            ok=False, path=[], moves=[], reach_in=req.climber.reach_in, total_cost=0.0,
            error=diagnostic, strategy="balanced",
            reasoning=f"Claude and the deterministic tools could not find a reachable route. {diagnostic}",
            alternatives=[], pushback=None, source="claude",
        )

    requested_strategy = str((final_action or {}).get("chosen_strategy", ""))
    selected_key = requested_strategy if requested_strategy in valid_candidates else None
    requested_path = (final_action or {}).get("path")
    if selected_key is None and requested_path:
        selected_key = next((k for k, v in valid_candidates.items() if v["path"] == requested_path), None)
    if selected_key is None:
        # If the model stops early or returns an invalid path, choose the best measured candidate.
        selected_key = min(valid_candidates, key=lambda k: valid_candidates[k]["total_cost"])
    chosen = valid_candidates[selected_key]

    alternatives = [
        PlanAlternative(**_candidate_summary(result, key))
        for key, result in valid_candidates.items()
        if key != selected_key and result["path"] != chosen["path"]
    ]
    reasoning = str((final_action or {}).get("reasoning") or "Selected from measured candidate routes based on move count, reach ratio, and hold quality.")
    pushback = _build_pushback(req, chosen, selected_key)
    return PlanResponse(
        ok=True, path=chosen["path"], moves=chosen["moves"], reach_in=req.climber.reach_in,
        total_cost=chosen["total_cost"], error=None, strategy=selected_key,
        reasoning=reasoning, alternatives=alternatives, pushback=pushback, source="claude",
    )


@app.get("/")
def health():
    return {"agent": "planner", "status": "ok", "llm": llm_available()}


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    if not llm_available():
        return _deterministic_response(req)
    return _llm_plan(req)
