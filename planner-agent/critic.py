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

from shared.llm import call_structured, llm_available
from shared.schemas import CritiqueRequest, CritiqueResponse

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
address your safety concern. Explain this decision briefly."""

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
    worst = max(req.plan.moves, key=lambda m: m.reach_ratio, default=None)
    final_round = req.round >= req.max_rounds

    if worst and not final_round:
        too_far = worst.reach_ratio > 0.9
        risky = worst.reach_ratio > 0.75 and (worst.to_type in POOR_HOLDS or worst.from_type in POOR_HOLDS)
        if too_far or risky:
            reason = (
                f"Move #{worst.from_id}->#{worst.to_id} is {worst.distance_in:.0f} in "
                f"({worst.reach_ratio:.0%} of max reach)"
                + (f" onto a {worst.to_type}" if worst.to_type in POOR_HOLDS else "")
                + ". Too stretched -- find an intermediate hold."
            )
            return CritiqueResponse(
                approved=False,
                reasoning=reason,
                avoid_moves=[[worst.from_id, worst.to_id]],
                source=f"rule-based ({why})",
            )
    return CritiqueResponse(
        approved=True,
        reasoning="Every move is within a comfortable stretch for this climber.",
        source=f"rule-based ({why})",
    )


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
        f"Proposed moves:\n{json.dumps([m.model_dump() for m in req.plan.moves], indent=1)}\n"
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
