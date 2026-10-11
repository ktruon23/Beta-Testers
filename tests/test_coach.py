import json

import pytest
from fastapi.testclient import TestClient

import coach
import database
from shared.schemas import Climber, CoachRequest, PlanResponse
from tests.conftest import ladder, move

client = TestClient(coach.app)
CLIMBER = Climber(height_in=66, wingspan_in=68)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "coach.db"))
    database.init()


def creq(moves, user_id="guest"):
    plan = PlanResponse(ok=bool(moves), path=[0] + [m.to_id for m in moves], moves=moves, reach_in=57.8)
    return CoachRequest(plan=plan, holds=ladder(5), climber=CLIMBER, user_id=user_id)


def test_rule_based_beta_one_step_per_move_and_finds_crux():
    moves = [move(0, 1, ratio=0.3), move(1, 2, ratio=0.95, dx=10, to_type="crimp"), move(2, 3, ratio=0.5)]
    out = coach.rule_based_beta(creq(moves), "test")
    assert [s.move for s in out.steps] == [1, 2, 3]
    assert "crux is move 2" in out.summary
    assert "near your max reach" in out.steps[1].instruction
    assert "right" in out.steps[1].instruction and "crimp" in out.steps[1].instruction


def test_beta_endpoint_no_route():
    out = client.post("/beta", json=creq([]).model_dump()).json()
    assert out["steps"] == [] and out["source"] == "none"


def test_beta_endpoint_without_llm():
    out = client.post("/beta", json=creq([move()]).model_dump()).json()
    assert len(out["steps"]) == 1 and out["source"].startswith("rule-based")


def test_feedback_then_history():
    m = move(to_type="sloper", ratio=0.8).model_dump()
    client.post("/feedback", json={"user_id": "noah", "move": m, "success": False, "notes": "slipped"})
    client.post("/feedback", json={"user_id": "noah", "move": m, "success": True})
    client.post("/feedback", json={"user_id": "someone-else", "move": m, "success": False})
    h = client.get("/history/noah").json()
    assert h["attempts"] == 2 and h["successes"] == 1
    assert h["recent_failures"][0]["notes"] == "slipped"


def test_history_filter_by_hold_type():
    database.record("noah", move(to_type="sloper"), False, "")
    database.record("noah", move(to_type="jug"), True, "")
    assert database.history("noah", "sloper")["attempts"] == 1
    assert database.history("nobody")["attempts"] == 0


def test_notes_cannot_inject_sql():
    database.record("noah", move(), False, "'); DROP TABLE attempts; --")
    assert database.history("noah")["attempts"] == 1


def test_executor_tools_and_bad_input():
    ex = coach.make_executor(creq([move(0, 1)]))
    assert json.loads(ex("get_move", {"move": 1}))["to_id"] == 1
    assert ex("get_move", {"move": 2}).startswith("Error")
    assert ex("get_hold", {"hold_id": 99}).startswith("Error")
    assert json.loads(ex("get_history", {}))["attempts"] == 0
    assert ex("nope", {}).startswith("Error")
