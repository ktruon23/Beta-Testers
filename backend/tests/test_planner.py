import pytest
from fastapi.testclient import TestClient

import planner
from backend.shared.schemas import Climber, Hold, PlanRequest
from backend.tests.conftest import ladder

client = TestClient(planner.app)


def req(holds, start=0, finish=None, ipp=0.2, climber=None, **kw):
    finish = holds[-1].id if finish is None else finish
    return PlanRequest(holds=holds, start_id=start, finish_id=finish, inches_per_pixel=ipp,
                       climber=climber or Climber(height_in=66, wingspan_in=68), **kw)


def test_straight_ladder_uses_every_hold_when_each_step_is_reachable_but_skips_are_not():
    # 100 px * 0.2 = 20 in per rung; two rungs = 40 in < 57.8 reach, three = 60 in > reach.
    r = planner.plan_route(req(ladder(5)))
    assert r["ok"]
    assert r["path"][0] == 0 and r["path"][-1] == 4
    assert all(m.distance_in <= 57.8 for m in r["moves"])


def test_moves_point_up_and_measure_in_inches():
    r = planner.plan_route(req(ladder(2)))
    (m,) = r["moves"]
    assert m.dy_in == pytest.approx(20.0)  # image y grows down, dy is flipped to "up is positive"
    assert m.dx_in == 0
    assert m.distance_in == pytest.approx(20.0)
    assert m.reach_ratio == pytest.approx(20.0 / 57.8, abs=1e-3)


def test_gap_bigger_than_reach_reports_closest_hold():
    holds = [Hold(id=0, x=0, y=1000), Hold(id=1, x=0, y=900), Hold(id=2, x=0, y=100)]
    r = planner.plan_route(req(holds))
    assert not r["ok"]
    assert r["closest_reachable_id"] == 1
    assert r["gap_distance_in"] == pytest.approx(160.0)
    assert "finish" in r["error"] and "#" not in r["error"]


def test_avoid_hold_forces_detour():
    holds = [
        Hold(id=0, x=100, y=1000), Hold(id=1, x=100, y=850),
        Hold(id=2, x=250, y=850), Hold(id=3, x=100, y=700),
    ]
    assert 1 in planner.plan_route(req(holds))["path"]
    detour = planner.plan_route(req(holds), avoid_holds=[1])
    assert detour["ok"] and 1 not in detour["path"] and 2 in detour["path"]


def test_avoid_move_is_direction_independent():
    holds = ladder(3)
    r = planner.plan_route(req(holds), avoid_moves=[[2, 0]])
    assert r["ok"] and r["path"] == [0, 1, 2]


def test_avoiding_start_or_finish_is_ignored():
    r = planner.plan_route(req(ladder(3)), avoid_holds=[0, 2])
    assert r["ok"] and r["path"][0] == 0 and r["path"][-1] == 2


def test_unknown_start_or_finish():
    r = planner.plan_route(req(ladder(3), finish=99))
    assert not r["ok"] and "not found" in r["error"]


def test_start_equals_finish_is_an_empty_route():
    r = planner.plan_route(req(ladder(3), start=1, finish=1))
    assert r["ok"] and r["path"] == [1] and r["moves"] == []


def test_find_gaps_lists_frontier():
    holds = [Hold(id=0, x=0, y=1000), Hold(id=1, x=0, y=900), Hold(id=2, x=0, y=100)]
    g = planner.find_gaps(req(holds))
    assert g["route_exists"] is False
    assert g["reachable_from_start"] == [0, 1]
    assert g["gaps"][0]["from_id"] == 1


def test_check_move():
    assert planner.check_move(req(ladder(3)), 0, 2)["reachable"] is True
    assert planner.check_move(req(ladder(5)), 0, 4)["reachable"] is False
    assert planner.check_move(req(ladder(3)), 0, 42)["ok"] is False


def test_plan_endpoint_without_llm_is_deterministic():
    body = req(ladder(4)).model_dump()
    out = client.post("/plan", json=body).json()
    assert out["ok"] and out["source"] == "deterministic" and out["strategy"] == "balanced"


def test_plan_endpoint_no_route():
    holds = [Hold(id=0, x=0, y=1000), Hold(id=1, x=0, y=100)]
    out = client.post("/plan", json=req(holds).model_dump()).json()
    assert out["ok"] is False and out["error"]
