"""End-to-end: the orchestrator talks to the real Planner/Critic/Coach apps in-process."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import coach
import critic
import orchestrator
import planner
from backend.shared.schemas import Hold

AGENTS = {
    orchestrator.PLANNER_URL: TestClient(planner.app),
    orchestrator.CRITIC_URL: TestClient(critic.app),
    orchestrator.COACH_URL: TestClient(coach.app),
}
CLIMBER = {"height_in": 66, "wingspan_in": 68}


@pytest.fixture
def wire(monkeypatch):
    """Route orchestrator.send() to in-process apps. Returns a dict to fake Vision / break agents."""
    state = {"vision": None, "down": set()}

    def fake_send(client, url, body):
        if url.startswith(orchestrator.VISION_URL):
            return state["vision"]
        for base, tc in AGENTS.items():
            if url.startswith(base):
                if base in state["down"]:
                    raise HTTPException(502, f"Agent at {url} failed")
                r = tc.post(url[len(base):], json=body)
                if r.status_code >= 400:
                    raise HTTPException(502, f"Agent at {url} failed: {r.text}")
                return r.json()
        raise AssertionError(url)

    monkeypatch.setattr(orchestrator, "send", fake_send)
    return state


api = TestClient(orchestrator.app)


def holds_json(holds):
    return [h if isinstance(h, dict) else h.model_dump() for h in holds]


def ladder(n=5, color="red", x=100):
    return [{"id": i, "x": x, "y": 1000 - i * 100, "type": "jug", "color": color} for i in range(n)]


def vision_response(holds, ipp=0.2):
    colors = {}
    for h in holds:
        colors[h["color"]] = colors.get(h["color"], 0) + 1
    return {"holds": holds, "image_width": 800, "image_height": 1200, "method": "fake",
            "inches_per_pixel": ipp, "scale_method": "median-hold-size", "colors": colors}


def test_solve_full_pipeline(wire):
    r = api.post("/solve", json={"holds": ladder(5), "inches_per_pixel": 0.2, "climber": CLIMBER})
    out = r.json()
    assert r.status_code == 200 and out["ok"]
    assert out["start_id"] == 0 and out["finish_id"] == 4
    assert out["plan"]["path"][0] == 0 and out["plan"]["path"][-1] == 4
    assert out["critic_approved"] and out["beta"]["steps"]
    assert [t["step"] for t in out["trace"]] == list(range(1, len(out["trace"]) + 1))
    assert out["orchestrator_source"].startswith("fixed")


@pytest.mark.parametrize("n", [0, 1])
def test_solve_needs_two_holds(wire, n):
    r = api.post("/solve", json={"holds": ladder(n), "inches_per_pixel": 0.2, "climber": CLIMBER})
    assert r.status_code == 400


def test_solve_unreachable_finish(wire):
    holds = [{"id": 0, "x": 0, "y": 1000}, {"id": 1, "x": 0, "y": 100}]
    out = api.post("/solve", json={"holds": holds, "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert out["ok"] is False and "No route" in out["status"]


def test_solve_bad_start_id_falls_back_to_lowest(wire):
    out = api.post("/solve", json={"holds": ladder(3), "start_id": 99, "inches_per_pixel": 0.2,
                                   "climber": CLIMBER}).json()
    assert out["start_id"] == 0


def test_all_holds_at_same_height_still_gives_distinct_start_and_finish(wire):
    holds = [{"id": i, "x": 100 + i * 100, "y": 500} for i in range(3)]
    out = api.post("/solve", json={"holds": holds, "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert out["start_id"] != out["finish_id"]
    assert out["ok"] and len(out["plan"]["moves"]) >= 1


def test_critic_constraints_reroute(wire):
    # The cheapest route 0->1->4 starts with a 53 in move (92% of reach); the rule-based
    # Critic rejects it and the Planner has to go around through 2.
    holds = [
        {"id": 0, "x": 100, "y": 1000, "type": "jug"},
        {"id": 1, "x": 100, "y": 735, "type": "jug"},   # 53 in from start = 92% of reach
        {"id": 2, "x": 300, "y": 900, "type": "jug"},   # detour holds
        {"id": 3, "x": 300, "y": 750, "type": "jug"},
        {"id": 4, "x": 100, "y": 600, "type": "jug"},
    ]
    out = api.post("/solve", json={"holds": holds, "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert out["ok"]
    assert out["rounds"] >= 2
    assert any(t["payload"].get("approved") is False for t in out["trace"])


def test_solve_photo_happy_path(wire):
    wire["vision"] = vision_response(ladder(4, "red") + ladder(2, "blue", x=500))
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "route_color": "Red "}).json()
    assert out["ok"]
    assert len(out["holds"]) == 4 and all(h["color"] == "red" for h in out["holds"])
    assert out["trace"][0]["receiver"] == "Vision"


@pytest.mark.parametrize("n", [0, 1])
def test_solve_photo_with_too_few_detected_holds_explains_the_photo(wire, n):
    wire["vision"] = vision_response(ladder(n))
    r = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER})
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert f"found {n} hold" in detail and "photo" in detail


def test_solve_photo_route_color_missing(wire):
    wire["vision"] = vision_response(ladder(5, "red"))
    r = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "route_color": "green"})
    assert r.status_code == 400 and "red" in r.json()["detail"]


def test_solve_photo_scale_override(wire):
    wire["vision"] = vision_response(ladder(5), ipp=5.0)  # absurd scale: nothing reachable
    bad = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER}).json()
    good = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "inches_per_pixel": 0.2}).json()
    assert bad["ok"] is False and good["ok"] is True


def test_coach_down_still_returns_the_route(wire):
    wire["down"].add(orchestrator.COACH_URL)
    r = api.post("/solve", json={"holds": ladder(4), "inches_per_pixel": 0.2, "climber": CLIMBER})
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] and out["plan"]["path"] and out["beta"]["steps"] == []


def test_critic_down_still_returns_the_route(wire):
    wire["down"].add(orchestrator.CRITIC_URL)
    r = api.post("/solve", json={"holds": ladder(4), "inches_per_pixel": 0.2, "climber": CLIMBER})
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] and out["critic_approved"] is False


def test_planner_down_is_a_502(wire):
    wire["down"].add(orchestrator.PLANNER_URL)
    r = api.post("/solve", json={"holds": ladder(4), "inches_per_pixel": 0.2, "climber": CLIMBER})
    assert r.status_code == 502
