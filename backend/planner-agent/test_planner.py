from planner import plan
from backend.shared.schemas import PlanRequest, Hold, Climber

holds = [
    Hold(id=0, x=100, y=500, type="jug"),
    Hold(id=1, x=100, y=400, type="jug"),
    Hold(id=2, x=100, y=300, type="sloper"),
    Hold(id=3, x=100, y=200, type="jug"),
]

req = PlanRequest(
    holds=holds,
    start_id=0,
    finish_id=3,
    inches_per_pixel=0.2,                       # 100 px = 20 in
    climber=Climber(height_in=66, wingspan_in=68),  # reach = 57.8 in
)

res = plan(req)
print(res.ok, res.path, res.total_cost, res.error)

# Simulate the Critic: avoid the sloper
req.avoid_holds = [2]
print(plan(req).path)