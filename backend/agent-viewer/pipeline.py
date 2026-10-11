"""
Step-by-step driver for the Beta Agents pipeline that reports what each agent is doing.

It calls the agent services directly (Vision -> Planner <-> Critic -> Coach) instead of
the Orchestrator's /solve, so every stage can be shown live as it happens. The round
logic mirrors orchestrator.solve(); keep the two in sync if you change one.

Every step yields an Event:
    stage   "detect" | "filter" | "holds" | "plan" | "critique" | "grade" | "coach" | "done" | "error"
    status  "running" | "done" | "failed"
    title   one line for a status header
    detail  the agent's "thinking": reasoning, strategy, alternatives, pushback...
    data    raw response, for anything the UI wants to draw

CLI (prints the live log in the terminal):
    cd agent-viewer
    python pipeline.py wall.jpg --height 68 --wingspan 70 --level beginner
    python pipeline.py wall.jpg --hold-types 3=crimp,5=sloper --default-hold jug
"""

from __future__ import annotations

import argparse
import base64
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

import httpx

VISION_URL = os.environ.get("VISION_URL", "http://localhost:8001")
PLANNER_URL = os.environ.get("PLANNER_URL", "http://localhost:8002")
CRITIC_URL = os.environ.get("CRITIC_URL", "http://localhost:8003")
COACH_URL = os.environ.get("COACH_URL", "http://localhost:8004")
TIMEOUT = httpx.Timeout(90.0)

# Mirrors shared/schemas.py: unclassified holds are assumed to be DEFAULT_HOLD_TYPE.
HOLD_TYPES = ("jug", "pinch", "pocket", "crimp", "sloper")
DEFAULT_HOLD_TYPE = "jug"


@dataclass
class Event:
    stage: str
    status: str
    title: str
    detail: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    round: int = 0


def _post(client: httpx.Client, url: str, body: dict) -> dict:
    r = client.post(url, json=body)
    if r.status_code >= 400:
        try:
            msg = r.json().get("detail", r.text)
        except ValueError:
            msg = r.text
        raise RuntimeError(f"{url} returned {r.status_code}: {msg}")
    return r.json()


# ---------------------------------------------------------------- explaining agents

def explain_detection(det: dict) -> list[str]:
    colors = ", ".join(f"{c} {n}" for c, n in det.get("colors", {}).items()) or "none"
    lines = [
        f"Found {len(det['holds'])} holds in a {det['image_width']}x{det['image_height']} photo ({det['method']}).",
        f"Colors: {colors}.",
    ]
    if det.get("inches_per_pixel"):
        wall_ft = det["image_height"] * det["inches_per_pixel"] / 12
        how = ("from the median hold size" if det.get("scale_method") == "median-hold-size"
               else "by assuming a 14 ft wall (too few holds to trust hold sizes)")
        lines.append(f"Scale: {det['inches_per_pixel']:.3f} in/px, estimated {how}, "
                     f"so the photo shows about {wall_ft:.1f} ft of wall.")
    return lines


def explain_plan(plan: dict) -> list[str]:
    if not plan["ok"]:
        return [f"No route: {plan.get('error') or 'unknown reason'}",
                *([f"Thinking: {plan['reasoning']}"] if plan.get("reasoning") else [])]

    moves = plan["moves"]
    hardest = max(moves, key=lambda m: m["reach_ratio"]) if moves else None
    lines = [
        f"Strategy: {plan.get('strategy', 'balanced')} ({plan.get('source', '?')}).",
        f"Route: {' -> '.join('#' + str(h) for h in plan['path'])} "
        f"({len(moves)} move{'s' if len(moves) != 1 else ''}, reach limit {plan['reach_in']:.0f} in).",
    ]
    if hardest:
        lines.append(f"Hardest move: #{hardest['from_id']} -> #{hardest['to_id']}, "
                     f"{hardest['distance_in']:.0f} in = {hardest['reach_ratio']:.0%} of reach "
                     f"onto {'an' if hardest['to_type'][0] in 'aeiou' else 'a'} {hardest['to_type']} hold.")
    if plan.get("reasoning"):
        lines.append(f"Thinking: {plan['reasoning']}")
    for alt in plan.get("alternatives", []):
        lines.append(f"Also considered '{alt['strategy']}': {len(alt['path']) - 1} moves, "
                     f"max reach {alt['max_reach_ratio']:.0%}, cost {alt['total_cost']:.2f}.")
    if plan.get("pushback"):
        pb = plan["pushback"]
        lines.append(f"Pushing back on the Critic: {pb['reason']} Proposal: {pb['proposal']}")
    return lines


def parse_hold_types(text: str) -> dict[int, str]:
    """"3=crimp, 5=sloper" -> {3: "crimp", 5: "sloper"}. Raises ValueError on bad input."""
    out: dict[int, str] = {}
    for part in filter(None, (p.strip() for p in (text or "").replace(";", ",").split(","))):
        hold_id, sep, kind = part.partition("=")
        kind = kind.strip().lower()
        if not sep or not hold_id.strip().isdigit() or kind not in HOLD_TYPES:
            raise ValueError(f"'{part}' should look like 3=crimp (types: {', '.join(HOLD_TYPES)})")
        out[int(hold_id)] = kind
    return out


def apply_hold_types(holds: list[dict], hold_types: dict[int, str],
                     default: str) -> tuple[list[dict], list[int]]:
    """Same rule as orchestrator.apply_hold_types: user's type, then Vision's, then the default."""
    out, assumed = [], []
    for h in holds:
        if h["id"] in hold_types:
            t = hold_types[h["id"]]
        elif (h.get("type") or "").lower() in HOLD_TYPES:
            t = h["type"].lower()
        else:
            t = default
            assumed.append(h["id"])
        out.append({**h, "type": t})
    return out, assumed


def explain_difficulty(diff: dict) -> list[str]:
    counts = ", ".join(f"{n} {t}{'s' if n != 1 else ''}" for t, n in diff.get("hold_counts", {}).items())
    lines = [f"{diff['grade']} ({diff['label']}, score {diff['score']}, {diff['source']}).",
             f"Holds on route: {counts}."]
    if diff.get("reasoning"):
        lines.append(f"Thinking: {diff['reasoning']}")
    if diff.get("assumed_holds"):
        lines.append(f"Assumed {diff['default_hold_type']}: " +
                     ", ".join(f"#{i}" for i in diff["assumed_holds"]) + ". Set their real types to sharpen the grade.")
    return lines


def explain_critique(crit: dict) -> list[str]:
    lines = [f"{'Approved' if crit['approved'] else 'Rejected'} ({crit['source']}).",
             f"Thinking: {crit['reasoning']}"]
    if not crit["approved"]:
        if crit.get("avoid_holds"):
            lines.append("Asks the Planner to avoid holds " + ", ".join(f"#{h}" for h in crit["avoid_holds"]) + ".")
        if crit.get("avoid_moves"):
            lines.append("Asks the Planner to avoid moves " +
                         ", ".join(f"#{a} -> #{b}" for a, b in crit["avoid_moves"]) + ".")
    return lines


# ---------------------------------------------------------------- the pipeline

def run(image_bytes: bytes, climber: dict, route_color: Optional[str] = None,
        start_id: Optional[int] = None, finish_id: Optional[int] = None,
        inches_per_pixel: Optional[float] = None, min_conf: float = 0.5,
        max_rounds: int = 4, detection: Optional[dict] = None,
        hold_types: Optional[dict[int, str]] = None,
        default_hold_type: str = DEFAULT_HOLD_TYPE) -> Iterator[Event]:
    """Run the agents, yielding an Event per step. Pass `detection` to reuse an earlier /detect result."""
    with httpx.Client(timeout=TIMEOUT) as client:
        try:
            # 1. Vision
            if detection is None:
                yield Event("detect", "running", "Vision is detecting holds in the photo...")
                det = _post(client, f"{VISION_URL}/detect", {
                    "image_b64": base64.b64encode(image_bytes).decode(), "min_conf": min_conf})
            else:
                det = detection
            yield Event("detect", "done", f"Vision found {len(det['holds'])} holds",
                        explain_detection(det), det)

            # 2. Route color
            holds = det["holds"]
            if route_color:
                holds = [h for h in holds if (h.get("color") or "").lower() == route_color.lower()]
                holds = [{**h, "id": i} for i, h in enumerate(holds)]  # renumber bottom -> top
                if len(holds) < 2:
                    yield Event("error", "failed", f"Only {len(holds)} {route_color} holds detected",
                                [f"Detected colors: {det.get('colors')}"])
                    return
                yield Event("filter", "done", f"Kept {len(holds)} {route_color} holds",
                            [f"Hold ids now run 0-{len(holds) - 1}, bottom to top."], {"holds": holds})

            # 2b. Hold types: the camera often can't see them, so fill in the user's or the default.
            hold_types = hold_types or {}
            bad = sorted(set(hold_types) - {h["id"] for h in holds})
            if bad:
                yield Event("error", "failed", f"Hold types set for holds {bad}, which aren't on this route",
                            [f"Valid ids: 0-{len(holds) - 1}."])
                return
            holds, assumed = apply_hold_types(holds, hold_types, default_hold_type)
            detail = [f"{len(assumed)} hold(s) with no known type are treated as {default_hold_type}s."]
            if hold_types:
                detail.append("You set: " + ", ".join(f"#{i} {t}" for i, t in sorted(hold_types.items())) + ".")
            yield Event("holds", "done", "Hold types set", detail, {"holds": holds, "assumed_holds": assumed})

            ipp = inches_per_pixel or det.get("inches_per_pixel")
            if not ipp:
                yield Event("error", "failed", "No scale available", ["Pass inches_per_pixel."])
                return

            ids = {h["id"] for h in holds}
            start = start_id if start_id in ids else max(holds, key=lambda h: h["y"])["id"]
            finish = finish_id if finish_id in ids else min(holds, key=lambda h: h["y"])["id"]

            # 3. Planner <-> Critic
            avoid_holds: set[int] = set()
            avoid_moves: list[list[int]] = []
            critic_feedback = None
            last_good = None
            approved = False
            status = ""

            for rnd in range(1, max_rounds + 1):
                note = (f" avoiding holds {sorted(avoid_holds)} and moves {avoid_moves}"
                        if (avoid_holds or avoid_moves) else "")
                yield Event("plan", "running", f"Round {rnd}: Planner is searching #{start} -> #{finish}{note}...",
                            round=rnd)
                plan = _post(client, f"{PLANNER_URL}/plan", {
                    "holds": holds, "start_id": start, "finish_id": finish,
                    "inches_per_pixel": ipp, "climber": climber,
                    "avoid_holds": sorted(avoid_holds), "avoid_moves": avoid_moves,
                    "critic_feedback": critic_feedback,
                })
                if not plan["ok"]:
                    yield Event("plan", "failed", f"Round {rnd}: Planner found no route",
                                explain_plan(plan), plan, rnd)
                    if last_good is None:
                        yield Event("error", "failed", "No route this climber can reach", explain_plan(plan), plan)
                        return
                    status = "The Critic's constraints left no route, so the previous one was kept."
                    break
                last_good = plan
                yield Event("plan", "done", f"Round {rnd}: Planner proposed {len(plan['moves'])} move{'s' if len(plan['moves']) != 1 else ''}",
                            explain_plan(plan), plan, rnd)

                yield Event("critique", "running", f"Round {rnd}: Critic is reviewing the route...", round=rnd)
                crit = _post(client, f"{CRITIC_URL}/critique", {
                    "plan": plan, "holds": holds, "climber": climber,
                    "start_id": start, "finish_id": finish,
                    "round": rnd, "max_rounds": max_rounds,
                    "planner_pushback": plan.get("pushback"),
                })
                verdict = "approved" if crit["approved"] else "rejected"
                yield Event("critique", "done", f"Round {rnd}: Critic {verdict} the route",
                            explain_critique(crit), crit, rnd)

                critic_feedback = {k: crit[k] for k in ("reasoning", "avoid_holds", "avoid_moves")}
                if crit["approved"]:
                    approved = True
                    status = f"Route approved by the Critic in round {rnd}."
                    break
                new_holds = set(crit["avoid_holds"]) - {start, finish} - avoid_holds
                new_moves = [m for m in crit["avoid_moves"] if m not in avoid_moves]
                if not new_holds and not new_moves:
                    status = "The Critic rejected the route but gave nothing new to change, so it was kept."
                    break
                avoid_holds |= new_holds
                avoid_moves += new_moves
            else:
                status = f"Hit the {max_rounds}-round limit; using the last route."

            # 4. Critic grades the final route from its hold types and reach
            difficulty = None
            yield Event("grade", "running", "Critic is grading the route's difficulty...")
            try:
                difficulty = _post(client, f"{CRITIC_URL}/difficulty", {
                    "plan": last_good, "holds": holds, "climber": climber,
                    "assumed_holds": assumed, "default_hold_type": default_hold_type})
                yield Event("grade", "done", f"Critic graded the route {difficulty['grade']}",
                            explain_difficulty(difficulty), difficulty)
            except (httpx.HTTPError, RuntimeError) as e:
                yield Event("grade", "failed", "Critic could not grade the route", [str(e)])

            # 5. Coach
            yield Event("coach", "running", "Coach is writing move-by-move beta...")
            try:
                beta = _post(client, f"{COACH_URL}/beta", {"plan": last_good, "holds": holds, "climber": climber})
                lines = [beta["summary"], *[f"{s['move']}. {s['instruction']}" for s in beta["steps"]],
                         *[f"Tip: {t}" for t in beta.get("tips", [])]]
                yield Event("coach", "done", "Coach wrote the beta", lines, beta)
            except (httpx.HTTPError, RuntimeError) as e:
                beta = None
                yield Event("coach", "failed", "Coach is not available", [str(e)])

            yield Event("done", "done", status, data={
                "plan": last_good, "holds": holds, "approved": approved, "beta": beta,
                "difficulty": difficulty,
                "start_id": start, "finish_id": finish, "detection": det})

        except (httpx.HTTPError, RuntimeError) as e:
            yield Event("error", "failed", "An agent call failed", [str(e)])


# ---------------------------------------------------------------- terminal mode

ICONS = {"running": "...", "done": "OK ", "failed": "!! "}
NAMES = {"detect": "VISION", "filter": "ORCH", "holds": "ORCH", "plan": "PLANNER", "critique": "CRITIC",
         "grade": "CRITIC",
         "coach": "COACH", "done": "DONE", "error": "ERROR"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Beta Agents and watch each agent think.")
    ap.add_argument("image")
    ap.add_argument("--color")
    ap.add_argument("--height", type=float, default=68)
    ap.add_argument("--wingspan", type=float, default=70)
    ap.add_argument("--level", default="intermediate", choices=["beginner", "intermediate", "expert"],
                    help="Climbing experience: beginner, intermediate or expert")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--hold-types", default="", type=parse_hold_types,
                    help="Known hold types by id, e.g. 3=crimp,5=sloper (ids after --color filtering)")
    ap.add_argument("--default-hold", default=DEFAULT_HOLD_TYPE, choices=HOLD_TYPES,
                    help="Type assumed for every hold you don't set (default: jug)")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img = f.read()
    climber = {"height_in": args.height, "wingspan_in": args.wingspan, "level": args.level}
    for ev in run(img, climber, route_color=args.color, max_rounds=args.rounds,
                  hold_types=args.hold_types, default_hold_type=args.default_hold):
        print(f"[{ICONS[ev.status]}] {NAMES[ev.stage]:<8} {ev.title}")
        for line in ev.detail:
            print(f"             {line}")
