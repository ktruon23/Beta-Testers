"""Hold types: user overrides, the default for holds the camera can't classify, and the Critic's grade."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import critic
from backend.shared.schemas import (
    DEFAULT_HOLD_TYPE,
    Climber,
    DifficultyRequest,
    PlanResponse,
    SolveRequest,
)
from tests.conftest import ladder, move
from tests.test_orchestrator import CLIMBER, api, wire  # noqa: F401  (wire is a fixture)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "agent-viewer"))
import pipeline  # noqa: E402

client = TestClient(critic.app)
CLIMB = Climber(height_in=66, wingspan_in=68)


def dreq(moves, holds=None, assumed=()):
    plan = PlanResponse(ok=True, path=[moves[0].from_id] + [m.to_id for m in moves] if moves else [],
                        moves=moves, reach_in=57.8)
    return DifficultyRequest(plan=plan, holds=holds or ladder(5), climber=CLIMB, assumed_holds=list(assumed))


def unknown_ladder(n=5):
    return [{"id": i, "x": 100, "y": 1000 - i * 100} for i in range(n)]  # type defaults to "unknown"


# ---------------- schemas

def test_default_hold_type_is_jug():
    assert DEFAULT_HOLD_TYPE == "jug"
    assert SolveRequest(holds=[], inches_per_pixel=1, climber=CLIMB).default_hold_type == "jug"


def test_hold_types_are_case_insensitive():
    req = SolveRequest(holds=[], inches_per_pixel=1, climber=CLIMB,
                       hold_types={"2": " Crimp"}, default_hold_type="SLOPER")
    assert req.hold_types == {2: "crimp"} and req.default_hold_type == "sloper"


@pytest.mark.parametrize("bad", [{"hold_types": {1: "volume"}}, {"default_hold_type": "edge"}])
def test_unknown_hold_type_rejected(bad):
    with pytest.raises(ValidationError):
        SolveRequest(holds=[], inches_per_pixel=1, climber=CLIMB, **bad)


# ---------------- Critic difficulty rating

def test_easy_jug_route_is_low_grade():
    out = critic.rate_difficulty(dreq([move(0, 1, ratio=0.4), move(1, 2, ratio=0.5)]))
    assert out.grade in ("V0", "V1") and out.label == "easy"


def test_same_moves_on_worse_holds_grade_harder():
    jugs = critic.rate_difficulty(dreq([move(0, 1, ratio=0.7), move(1, 2, ratio=0.7)]))
    crimps = critic.rate_difficulty(dreq(
        [move(0, 1, ratio=0.7, to_type="crimp"), move(1, 2, ratio=0.7, from_type="crimp", to_type="sloper")],
        holds=ladder(5, types=["jug", "crimp", "sloper", "jug", "jug"])))
    assert int(crimps.grade[1:]) > int(jugs.grade[1:])
    assert crimps.hardest_move.to_type == "sloper"
    assert crimps.hold_counts == {"jug": 1, "crimp": 1, "sloper": 1}


def test_longer_reach_grades_harder():
    short = critic.rate_difficulty(dreq([move(ratio=0.3, to_type="crimp")]))
    long = critic.rate_difficulty(dreq([move(ratio=1.0, to_type="crimp")]))
    assert long.score > short.score


def test_assumed_holds_are_reported():
    out = critic.rate_difficulty(dreq([move(0, 1), move(1, 2)], assumed=[1, 2, 4]))
    assert out.assumed_holds == [1, 2]  # only holds on the route
    assert "guessed" in out.reasoning


def test_difficulty_endpoint_without_llm():
    r = client.post("/difficulty", json=dreq([move(ratio=0.9, to_type="sloper")]).model_dump())
    assert r.status_code == 200
    body = r.json()
    assert body["grade"].startswith("V") and body["source"] == "rule-based"


def test_difficulty_endpoint_empty_plan():
    body = DifficultyRequest(plan=PlanResponse(ok=False, reach_in=57.8), holds=ladder(2), climber=CLIMB)
    assert client.post("/difficulty", json=body.model_dump()).json()["grade"] == "V0"


# ---------------- Orchestrator

def test_solve_assumes_default_for_unknown_holds(wire):  # noqa: F811
    out = api.post("/solve", json={"holds": unknown_ladder(), "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert out["ok"]
    assert {h["type"] for h in out["holds"]} == {"jug"}
    assert out["difficulty"]["assumed_holds"] == out["plan"]["path"]
    assert out["difficulty"]["grade"].startswith("V")


def test_solve_user_hold_types_win(wire):  # noqa: F811
    out = api.post("/solve", json={"holds": unknown_ladder(), "inches_per_pixel": 0.2, "climber": CLIMBER,
                                   "hold_types": {"2": "crimp"}, "default_hold_type": "pinch"}).json()
    types = {h["id"]: h["type"] for h in out["holds"]}
    assert types[2] == "crimp" and types[0] == "pinch"
    assert 2 not in out["difficulty"]["assumed_holds"]
    assert any("You marked" in t["summary"] and "as a crimp" in t["summary"] for t in out["trace"])


def test_solve_keeps_vision_hold_types(wire):  # noqa: F811
    holds = unknown_ladder()
    holds[1]["type"] = "sloper"
    out = api.post("/solve", json={"holds": holds, "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert {h["id"]: h["type"] for h in out["holds"]}[1] == "sloper"
    assert 1 not in out["difficulty"]["assumed_holds"]


def test_harder_holds_give_harder_grade(wire):  # noqa: F811
    body = {"holds": unknown_ladder(), "inches_per_pixel": 0.2, "climber": CLIMBER}
    easy = api.post("/solve", json=body).json()["difficulty"]
    hard = api.post("/solve", json={**body, "default_hold_type": "sloper"}).json()["difficulty"]
    assert hard["score"] > easy["score"]


def test_solve_hold_types_for_missing_hold_is_400(wire):  # noqa: F811
    r = api.post("/solve", json={"holds": unknown_ladder(3), "inches_per_pixel": 0.2, "climber": CLIMBER,
                                 "hold_types": {"7": "crimp"}})
    assert r.status_code == 400 and "7" in r.json()["detail"]


def test_solve_still_works_when_critic_cannot_grade(wire, monkeypatch):  # noqa: F811
    import orchestrator
    real = orchestrator.send

    def no_grade(client, url, body):
        if url.endswith("/difficulty"):
            from fastapi import HTTPException
            raise HTTPException(502, "down")
        return real(client, url, body)

    monkeypatch.setattr(orchestrator, "send", no_grade)
    out = api.post("/solve", json={"holds": unknown_ladder(), "inches_per_pixel": 0.2, "climber": CLIMBER}).json()
    assert out["ok"] and out["difficulty"] is None


def test_solve_photo_passes_hold_types(wire):  # noqa: F811
    from tests.test_orchestrator import vision_response
    holds = [{**h, "color": "red"} for h in unknown_ladder(4)]
    wire["vision"] = vision_response(holds)
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "hold_types": {"1": "pocket"}}).json()
    assert out["ok"] and out["holds"][1]["type"] == "pocket" and out["holds"][0]["type"] == "jug"


# ---------------- viewer CLI parsing

def test_parse_hold_types():
    assert pipeline.parse_hold_types("3=crimp, 5=Sloper;7=jug") == {3: "crimp", 5: "sloper", 7: "jug"}
    assert pipeline.parse_hold_types("") == {}


@pytest.mark.parametrize("bad", ["3", "x=crimp", "3=volume"])
def test_parse_hold_types_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        pipeline.parse_hold_types(bad)


def test_pipeline_apply_hold_types():
    holds, assumed = pipeline.apply_hold_types(
        [{"id": 0, "type": "unknown"}, {"id": 1, "type": "crimp"}, {"id": 2}], {2: "pinch"}, "jug")
    assert [h["type"] for h in holds] == ["jug", "crimp", "pinch"] and assumed == [0]
