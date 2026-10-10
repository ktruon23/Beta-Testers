import heapq
import math

from fastapi import FastAPI

from shared.schemas import Move, PlanRequest, PlanResponse

app = FastAPI(title = "Planner Agent")

HOLD_PENALTY = {"jug": 0, "pinch": 1, "pocket": 1.5, "crimp": 1.5, "sloper": 2, "unknown": 0.5}

def make_move(a, b, ipp, reach):
    dx = (b.x - a.x) * ipp
    dy = -(b.y - a.y) * ipp
    dist = math.hypot(dx, dy)
    return Move(
        from_id=a.id,
        to_id=b.id,
        distance_in=round(dist, 1),
        reach_ratio=round(dist / reach, 3),
        dx_in=round(dx, 1),
        dy_in=round(dy, 1),
        from_type=a.type,
        to_type=b.type,        
    )

def move_cost(move):
    return 1 + move.reach_ratio + HOLD_PENALTY.get(move.to_type, 0.5)

def plan_route(req: PlanRequest) -> PlanResponse:
    reach = req.climber.reach_in
    ipp = req.inches_per_pixel

    # 1. Apply the Critic's constraints (never remove start or finish)
    keep = {req.start_id, req.finish_id}
    blocked_holds = set(req.avoid_holds) - keep
    blocked_moves = {frozenset(p) for p in req.avoid_moves if len(p) == 2}
    holds = {h.id: h for h in req.holds if h.id not in blocked_holds}

    if req.start_id not in holds or req.finish_id not in holds:
        return PlanResponse(ok=False, reach_in=round(reach, 1), error="Start or finish hold is missing.")
    if req.start_id == req.finish_id:
        return PlanResponse(ok=False, reach_in=round(reach, 1), error="Start and finish are the same hold.")

    # 2. Dijkstra: always expand the cheapest hold found so far
    best = {req.start_id: 0.0}         # hold id -> cheapest cost to get there
    came_from = {req.start_id: None}   # hold id -> (previous hold id, the move)
    queue = [(0.0, req.start_id)]
    done = set()

    while queue:
        cost, current = heapq.heappop(queue)
        if current in done:
            continue
        done.add(current)
        if current == req.finish_id:
            break

        for other_id, other in holds.items():
            if other_id in done:
                continue
            if frozenset((current, other_id)) in blocked_moves:
                continue
            move = make_move(holds[current], other, ipp, reach)
            if move.distance_in > reach:  # too far to reach
                continue
            new_cost = cost + move_cost(move)
            if new_cost < best.get(other_id, math.inf):
                best[other_id] = new_cost
                came_from[other_id] = (current, move)
                heapq.heappush(queue, (new_cost, other_id))

    # 3. No route?
    if req.finish_id not in done:
        finish = holds[req.finish_id]
        gap = min(make_move(holds[h], finish, ipp, reach).distance_in for h in done)
        return PlanResponse(
            ok=False,
            reach_in=round(reach, 1),
            error=f"No route found. Closest we got is {gap:.0f} in from the finish; max move is {reach:.0f} in.",
        )

    # 4. Walk backwards from the finish to rebuild the path
    path, moves = [req.finish_id], []
    node = req.finish_id
    while came_from[node] is not None:
        prev, move = came_from[node]
        moves.append(move)
        path.append(prev)
        node = prev
    path.reverse()
    moves.reverse()

    return PlanResponse(
        ok=True,
        path=path,
        moves=moves,
        reach_in=round(reach, 1),
        total_cost=round(best[req.finish_id], 2),
    )


@app.get("/")
def health():
    return {"agent": "planner", "status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    return plan_route(req)
