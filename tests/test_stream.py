"""/solve_photo/stream sends each agent message as it is logged, then the final result."""

import json

import orchestrator
from .test_orchestrator import CLIMBER, api, ladder, vision_response, wire  # noqa: F401


def events(body):
    with api.stream("POST", "/solve_photo/stream", json=body) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


def test_stream_sends_trace_then_result(wire):  # noqa: F811
    wire["vision"] = vision_response(ladder(4, "red") + ladder(2, "blue", x=500))
    out = events({"image_b64": "x", "climber": CLIMBER, "route_color": "red"})
    traces = [e for e in out if e["type"] == "trace"]
    assert out[-1]["type"] == "result" and out[-1]["result"]["ok"]
    assert traces[0]["entry"]["receiver"] == "Vision"
    assert [e["entry"]["step"] for e in traces] == list(range(1, len(traces) + 1))
    assert len(traces) == len(out[-1]["result"]["trace"])
    assert getattr(orchestrator._live, "listener", None) is None
    # The Planner's proposal carries a sketch: the route plus its hardest move, in image pixels.
    proposal = next(e["entry"] for e in traces if e["entry"]["sender"] == "Planner")
    draw = proposal["payload"]["draw"]
    assert len(draw["path"]) >= 2 and draw["path"][0] == [100, 1000]
    assert draw["highlights"][0]["label"].startswith("move ")


def test_stream_reports_errors(wire):  # noqa: F811
    wire["vision"] = vision_response(ladder(1))
    out = events({"image_b64": "x", "climber": CLIMBER})
    assert out[-1]["type"] == "error" and "photo" in out[-1]["detail"].lower()


def test_picked_start_and_finish_snap_to_nearest_hold(wire):  # noqa: F811
    wire["vision"] = vision_response(ladder(5, "red"))  # y = 1000, 900, ... 600 at x = 100
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "route_color": "red",
                                         "start_point": [104, 905], "finish_point": [98, 702]}).json()
    assert out["ok"] and out["start_id"] == 1 and out["finish_id"] == 3
    assert any(t["summary"].startswith("You picked") for t in out["trace"])


def test_vision_message_is_plain_language(wire):  # noqa: F811
    wire["vision"] = vision_response(ladder(4, "red") + ladder(2, "blue", x=500))
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER}).json()
    vision = next(t["summary"] for t in out["trace"] if t["sender"] == "Vision")
    assert vision == "I found 6 holds: 4 red and 2 blue."


def test_grips_set_on_the_photo_change_hold_types(wire):  # noqa: F811
    wire["vision"] = vision_response(ladder(5, "red") + ladder(2, "blue", x=500))
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "route_color": "red", "grips": [
        {"x": 102, "y": 798, "type": "Sloper"},   # on red hold 2
        {"x": 500, "y": 1000, "type": "crimp"},   # a blue hold: not on this route, ignored
    ]}).json()
    types = {(h["x"], h["y"]): h["type"] for h in out["holds"]}
    assert types[(100, 800)] == "sloper" and "crimp" not in types.values()
    note = next(t for t in out["trace"] if t["summary"].startswith("You marked"))
    assert note["payload"]["draw"]["marks"] == [{"at": [100, 800], "label": "sloper"}]
