"""
CRITIC AGENT  -- uses Claude
Input:  a proposed route (with real distances, reach ratios, hold types)
Output: approve it, or reject it with SPECIFIC constraints for the Planner

The Critic plays an experienced climbing coach. It can't invent distances --
every number it sees was calculated by the Planner. Its job is judgment:
"that move is technically in reach, but launching off a sloper at 95% of your
span is a fall waiting to happen -- avoid that move."

If no API key is set (or the API call fails), it falls back to simple rules
so the system still runs. The response's `source` field says which one ran.
"""

import json

from fastapi import FastAPI

from backend.shared.llm import call_structured, llm_available
from backend.shared.schemas import (
    CritiqueRequest,
    CritiqueResponse,
    DifficultyRequest,
    Move,
    RouteDifficulty,
)

app = FastAPI(title="Critic Agent")

SYSTEM = """You are an experienced bouldering coach reviewing a route that a \
planning algorithm proposed for one specific climber.

All distances were calculated precisely from the wall photo -- trust the numbers, \
don't recompute them. reach_ratio = move distance / this climber's max reach \
(1.0 means fully stretched). dy_in > 0 means the move goes up.

Approve the route if a reasonable climber of this size could do it. Reject it only \
for a real problem, for example:
- a move above ~0.9 reach_ratio onto or off a poor hold (sloper, crimp, pocket)
- several near-max moves in a row with no rest
- a big move that goes sideways or down when an easier option likely exists

When you reject, give the planner concrete constraints: hold ids to avoid and/or \
specific moves (pairs of hold ids) to avoid. Never ask to avoid the start or finish \
hold. Be decisive and brief.

On the final round, approve unless the route is genuinely dangerous.

If the Planner includes a pushback proposal, evaluate its numeric evidence and
alternative. Accept the counter-proposal when it addresses the risky move while
preserving a reasonable route. Hold your position if the evidence does not
address your safety concern. Explain this decision briefly.

How to write reasoning: talk to the climber in plain, friendly language, like a \
coach at the gym. Refer to moves by their number (move 1 is the first move) and to \
holds by type ("the sloper"), never by hold id numbers. Round distances to whole \
inches. No jargon like reach_ratio."""

TOOL = {
    "name": "submit_verdict",
    "description": "Submit your verdict on the proposed route.",
    "input_schema": {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "reasoning": {
                "type": "string",
                "description": "1-3 sentences a climber would understand.",
            },
            "avoid_holds": {"type": "array", "items": {"type": "integer"}},
            "avoid_moves": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
        },
        "required": ["approved", "reasoning"],
    },
}

POOR_HOLDS = {"sloper", "crimp", "pocket"}


def rule_based(req: CritiqueRequest, why: str) -> CritiqueResponse:
    """Backup critic: flags the single worst stretched move."""
    k, worst = max(enumerate(req.plan.moves, 1), key=lambda km: km[1].reach_ratio, default=(0, None))
    final_round = req.round >= req.max_rounds

    if worst and not final_round:
        too_far = worst.reach_ratio > 0.9
        risky = worst.reach_ratio > 0.75 and (worst.to_type in POOR_HOLDS or worst.from_type in POOR_HOLDS)
        if too_far or risky:
            reason = (
                f"Move {k} is a {worst.distance_in:.0f} in reach"
                + (f" onto a {worst.to_type}" if worst.to_type in POOR_HOLDS else "")
                + f", {worst.reach_ratio:.0%} of what you can reach. That's a big stretch, "
                "so let's find a hold in between."
            )
            return CritiqueResponse(
                approved=False,
                reasoning=reason,
                avoid_moves=[[worst.from_id, worst.to_id]],
                source=f"rule-based ({why})",
            )
    return CritiqueResponse(
        approved=True,
        reasoning="Every move is a comfortable reach for you.",
        source=f"rule-based ({why})",
    )


# ---------------- difficulty rating ----------------

# How hard each hold type is to hold onto, 1 (jug) to 4 (sloper).
HOLD_DIFFICULTY = {"jug": 1.0, "pinch": 2.5, "pocket": 3.0, "crimp": 3.5, "sloper": 4.0}
LABELS = [(1, "easy"), (3, "moderate"), (5, "hard"), (99, "very hard")]  # (max V grade, label)


def move_difficulty(m: Move) -> float:
    """Landing hold counts most, the hold you leave from less; longer reaches make both harder."""
    hold = 0.65 * HOLD_DIFFICULTY.get(m.to_type, 1.0) + 0.35 * HOLD_DIFFICULTY.get(m.from_type, 1.0)
    return hold * (0.6 + 0.8 * m.reach_ratio)  # 50% reach -> x1.0, full reach -> x1.4


def rate_difficulty(req: DifficultyRequest) -> RouteDifficulty:
    """Deterministic grade from hold types and reach. The crux move dominates, like a real grade."""
    moves = req.plan.moves
    types = {h.id: h.type for h in req.holds}
    on_route = [types.get(i, req.default_hold_type) for i in req.plan.path]
    counts = {t: on_route.count(t) for t in dict.fromkeys(on_route)}
    assumed = [i for i in req.plan.path if i in set(req.assumed_holds)]

    if not moves:
        return RouteDifficulty(grade="V0", score=0.0, label="easy", hold_counts=counts,
                               assumed_holds=assumed, default_hold_type=req.default_hold_type,
                               reasoning="There are no moves to rate.")

    scores = sorted((move_difficulty(m) for m in moves), reverse=True)
    top = scores[:3]
    route_score = 0.6 * scores[0] + 0.4 * sum(top) / len(top)
    v = max(0, min(10, round((route_score - 0.9) * 1.3)))
    label = next(name for cap, name in LABELS if v <= cap)
    crux = max(moves, key=move_difficulty)

    k = moves.index(crux) + 1
    reasoning = (f"The hardest part is move {k}, from a {crux.from_type} to a {crux.to_type}, "
                 f"{crux.reach_ratio:.0%} of your reach.")
    if assumed:
        reasoning += (f" I couldn't tell what {len(assumed)} of the {len(req.plan.path)} holds "
                      f"{'is' if len(assumed) == 1 else 'are'}, "
                      f"so I guessed {req.default_hold_type}s; the grade may change once we know.")
    return RouteDifficulty(grade=f"V{v}", score=round(route_score, 2), label=label, hardest_move=crux,
                           hold_counts=counts, assumed_holds=assumed,
                           default_hold_type=req.default_hold_type, reasoning=reasoning)


DIFFICULTY_SYSTEM = """You are an experienced bouldering coach grading a route for one \
specific climber. A rule-based grade was computed from the hold types (jug easiest, then \
pinch, pocket, crimp, sloper hardest) and how far each move stretches this climber. \
You may move the grade by at most one V grade if the holds, sequence or climber's level \
justify it. Explain the grade in 1-2 sentences a climber would understand, naming the crux. \
If some hold types were assumed rather than given, say the grade could change once they're known.

How to write reasoning: talk to the climber in plain, friendly language, like a \
coach at the gym. Refer to moves by their number (move 1 is the first move) and to \
holds by type ("the sloper"), never by hold id numbers. Round distances to whole \
inches. No jargon like reach_ratio."""

DIFFICULTY_TOOL = {
    "name": "submit_grade",
    "description": "Submit the difficulty grade for this route.",
    "input_schema": {
        "type": "object",
        "properties": {
            "adjust": {"type": "integer", "enum": [-1, 0, 1],
                       "description": "Change to the rule-based V grade."},
            "reasoning": {"type": "string"},
        },
        "required": ["adjust", "reasoning"],
    },
}


@app.post("/difficulty", response_model=RouteDifficulty)
def difficulty(req: DifficultyRequest):
    rated = rate_difficulty(req)
    if not req.plan.moves or not llm_available():
        return rated

    user = (
        f"Climber: {req.climber.height_in:.0f} in tall, {req.climber.wingspan_in:.0f} in wingspan, "
        f"level {req.climber.level}.\n"
        f"Rule-based grade: {rated.grade} (score {rated.score}).\n"
        f"Hold types on route: {json.dumps(rated.hold_counts)}. "
        f"Assumed (default {req.default_hold_type}): {rated.assumed_holds}\n"
        f"Moves:\n{json.dumps([{'move': i, **m.model_dump()} for i, m in enumerate(req.plan.moves, 1)], indent=1)}"
    )
    try:
        out = call_structured(DIFFICULTY_SYSTEM, user, DIFFICULTY_TOOL)
    except Exception as e:
        rated.source = f"rule-based (Claude call failed: {type(e).__name__})"
        return rated

    adjust = out.get("adjust") if out.get("adjust") in (-1, 0, 1) else 0
    v = max(0, min(10, int(rated.grade[1:]) + adjust))
    rated.grade = f"V{v}"
    rated.label = next(name for cap, name in LABELS if v <= cap)
    rated.reasoning = str(out.get("reasoning") or rated.reasoning)
    rated.source = "claude"
    return rated


@app.get("/")
def health():
    return {"agent": "critic", "status": "ok", "llm": llm_available()}


@app.post("/critique", response_model=CritiqueResponse)
def critique(req: CritiqueRequest):
    if not req.plan.ok or not req.plan.moves:
        return CritiqueResponse(approved=False, reasoning="There is no route to review.", source="rule-based")

    if not llm_available():
        return rule_based(req, "no API key")

    hold_types = {h.id: h.type for h in req.holds if h.id in req.plan.path}
    user = (
        f"Round {req.round} of {req.max_rounds}.\n"
        f"Climber: {req.climber.height_in:.0f} in tall, {req.climber.wingspan_in:.0f} in wingspan, "
        f"max single move {req.climber.reach_in:.0f} in.\n"
        f"Start hold: #{req.start_id}. Finish hold: #{req.finish_id}.\n"
        f"Holds on route (id -> type): {json.dumps(hold_types)}\n"
        f"Proposed moves:\n{json.dumps([{'move': i, **m.model_dump()} for i, m in enumerate(req.plan.moves, 1)], indent=1)}\n"
        f"Planner pushback/counter-proposal, if any: {json.dumps(req.planner_pushback.model_dump() if req.planner_pushback else None)}"
    )
    try:
        out = call_structured(SYSTEM, user, TOOL)
    except Exception as e:  # bad key, network, rate limit... keep the demo alive
        return rule_based(req, f"Claude call failed: {type(e).__name__}")

    # The model can return junk ids (strings, triples, null); keep only well-formed ones.
    protected = {req.start_id, req.finish_id}
    holds = out.get("avoid_holds") or []
    moves = out.get("avoid_moves") or []
    return CritiqueResponse(
        approved=bool(out.get("approved")),
        reasoning=str(out.get("reasoning", "")),
        avoid_holds=[h for h in holds if type(h) is int and h not in protected],
        avoid_moves=[list(p) for p in moves
                     if isinstance(p, (list, tuple)) and len(p) == 2 and all(type(x) is int for x in p)],
        source="claude",
    )
