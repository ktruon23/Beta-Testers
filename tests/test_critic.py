from unittest.mock import patch

from fastapi.testclient import TestClient

import critic
from backend.shared.schemas import Climber, CritiqueRequest, PlanResponse
from tests.conftest import ladder, move

client = TestClient(critic.app)
CLIMBER = Climber(height_in=66, wingspan_in=68)


def creq(moves, rnd=1, max_rounds=4):
    plan = PlanResponse(ok=True, path=[0] + [m.to_id for m in moves], moves=moves, reach_in=57.8)
    return CritiqueRequest(plan=plan, holds=ladder(5), climber=CLIMBER, start_id=0, finish_id=4,
                           round=rnd, max_rounds=max_rounds)


def test_comfortable_route_approved():
    out = critic.rule_based(creq([move(ratio=0.4), move(1, 2, ratio=0.5)]), "test")
    assert out.approved and out.avoid_moves == []


def test_overstretched_move_rejected_with_constraint():
    out = critic.rule_based(creq([move(ratio=0.4), move(1, 2, ratio=0.95)]), "test")
    assert not out.approved and out.avoid_moves == [[1, 2]]


def test_stretched_move_onto_poor_hold_rejected():
    out = critic.rule_based(creq([move(0, 1, ratio=0.8, to_type="sloper")]), "test")
    assert not out.approved and "sloper" in out.reasoning


def test_final_round_always_approves():
    out = critic.rule_based(creq([move(ratio=0.99)], rnd=4, max_rounds=4), "test")
    assert out.approved


def test_endpoint_rejects_empty_plan():
    body = creq([]).model_dump()
    body["plan"]["ok"] = False
    out = client.post("/critique", json=body).json()
    assert out["approved"] is False


def test_endpoint_uses_rules_without_llm():
    out = client.post("/critique", json=creq([move(ratio=0.3)]).model_dump()).json()
    assert out["approved"] and out["source"].startswith("rule-based")


def test_malformed_llm_constraints_are_dropped_not_crashing():
    # The LLM is free text underneath; it can return strings, triples or the start hold.
    bad = {"approved": False, "reasoning": "nope",
           "avoid_holds": [0, "3", 2, None], "avoid_moves": [[1, 2], ["1", "2"], [1, 2, 3], "x"]}
    with patch.object(critic, "llm_available", return_value=True), \
         patch.object(critic, "call_structured", return_value=bad):
        r = client.post("/critique", json=creq([move(ratio=0.3)]).model_dump())
    assert r.status_code == 200
    out = r.json()
    assert out["avoid_holds"] == [2]          # start (0) protected, junk dropped
    assert out["avoid_moves"] == [[1, 2]]


def test_llm_failure_falls_back_to_rules():
    with patch.object(critic, "llm_available", return_value=True), \
         patch.object(critic, "call_structured", side_effect=RuntimeError("down")):
        out = client.post("/critique", json=creq([move(ratio=0.3)]).model_dump()).json()
    assert out["approved"] and "RuntimeError" in out["source"]
