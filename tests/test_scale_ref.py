"""User scale calibration: two taps on the photo plus the real distance between them (scale_ref)."""

import pytest

from backend.shared.schemas import ScaleRef
from tests.test_orchestrator import CLIMBER, api, ladder, vision_response, wire  # noqa: F401  (wire is a fixture)


def test_scale_ref_inches_per_pixel():
    ref = ScaleRef(a=(0, 0), b=(300, 400), inches=168)  # 500 px apart
    assert ref.pixels == 500 and ref.inches_per_pixel == pytest.approx(0.336)


def scale_ref(inches=100, a=(100, 1000), b=(100, 500)):
    return {"a": list(a), "b": list(b), "inches": inches}  # 500 px -> 0.2 in/px by default


def test_solve_photo_scale_ref_overrides_vision(wire):
    wire["vision"] = vision_response(ladder(5), ipp=5.0)  # absurd hold-size scale: nothing reachable
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "scale_ref": scale_ref()}).json()
    assert out["ok"] is True
    assert sum(m["distance_in"] for m in out["plan"]["moves"]) == pytest.approx(80)  # 400 px straight up
    said = [t for t in out["trace"] if "measurement" in t["summary"]]
    assert len(said) == 1 and not any(c.isdigit() for c in said[0]["summary"])


def test_solve_photo_scale_ref_wins_over_inches_per_pixel(wire):
    wire["vision"] = vision_response(ladder(5), ipp=5.0)
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "inches_per_pixel": 5.0,
                                         "scale_ref": scale_ref()}).json()
    assert out["ok"] is True


def test_solve_photo_without_scale_ref_uses_vision_estimate(wire):
    wire["vision"] = vision_response(ladder(5), ipp=0.3)
    out = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER}).json()
    assert out["ok"] is True
    assert sum(m["distance_in"] for m in out["plan"]["moves"]) == pytest.approx(120)  # 400 px straight up
    assert not any("measurement" in t["summary"] for t in out["trace"])


@pytest.mark.parametrize("ref", [
    scale_ref(inches=0),
    scale_ref(inches=-12),
    scale_ref(a=(100, 500), b=(105, 510)),   # taps ~11 px apart: too short to measure from
    {"a": [0, 0], "inches": 100},            # missing a point
])
def test_solve_photo_rejects_bad_scale_ref(wire, ref):
    wire["vision"] = vision_response(ladder(5))
    r = api.post("/solve_photo", json={"image_b64": "x", "climber": CLIMBER, "scale_ref": ref})
    assert r.status_code == 422
