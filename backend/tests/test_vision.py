import base64
import sys
import types

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import color_filter
import main as vision
import object_detection
import scale


# ---------------------------------------------------------------- helpers

def rle_encode(mask: np.ndarray) -> dict:
    """COCO compressed RLE, the format Roboflow returns (inverse of color_filter.decode_rle)."""
    h, w = mask.shape
    flat = mask.T.flatten()  # column-major
    counts, cur, run = [], 0, 0
    for v in flat:
        if v != cur:
            counts.append(run)
            cur, run = v, 0
        run += 1
    counts.append(run)
    out = []
    for i, x in enumerate(counts):
        x = int(x)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return {"size": [h, w], "counts": "".join(out)}


def pred(x, y, size=40, conf=0.9, class_id=1, color="red"):
    return {"x": x, "y": y, "width": size, "height": size, "confidence": conf,
            "class_id": class_id, "color": (color, (0, 0, 255), 1.0)}


def wall_with_hold(bgr, wall=(128, 128, 128)):
    img = np.full((200, 200, 3), wall, np.uint8)
    img[80:120, 80:120] = bgr
    return img


# ---------------------------------------------------------------- color

def test_rle_round_trip():
    mask = np.zeros((30, 20), np.uint8)
    mask[5:12, 3:9] = 1
    mask[20:25, 15:20] = 1
    assert np.array_equal(color_filter.decode_rle(rle_encode(mask)), mask)


@pytest.mark.parametrize("bgr, name", [
    ((0, 0, 220), "red"), ((220, 0, 0), "blue"), ((0, 200, 0), "green"),
    ((0, 220, 220), "yellow"), ((240, 240, 240), "white"), ((10, 10, 10), "black"),
])
def test_classify_solid_hold_on_gray_wall(bgr, name):
    got, _, conf = color_filter.classify_color(wall_with_hold(bgr), (80, 80, 119, 119))
    assert got == name and conf > 0.5


def test_classify_uses_mask_when_given():
    img = np.full((200, 200, 3), (0, 0, 220), np.uint8)   # red everywhere...
    img[90:110, 90:110] = (220, 0, 0)                      # ...except a blue hold
    mask = np.zeros((200, 200), np.uint8)
    mask[90:110, 90:110] = 1
    assert color_filter.classify_color(img, (80, 80, 119, 119), rle_encode(mask))[0] == "blue"


def test_classify_mask_from_resized_image():
    img = wall_with_hold((220, 0, 0))
    small = np.zeros((100, 100), np.uint8)
    small[40:60, 40:60] = 1                                # same hold at half resolution
    assert color_filter.classify_color(img, (80, 80, 119, 119), rle_encode(small))[0] == "blue"


@pytest.mark.parametrize("box", [(150, 150, 140, 140), (-50, -50, -10, -10), (500, 500, 600, 600)])
def test_classify_empty_or_offscreen_box(box):
    assert color_filter.classify_color(wall_with_hold((0, 0, 220)), box) == ("unknown", (0, 0, 0), 0.0)


# ---------------------------------------------------------------- scale

def test_scale_from_median_hold_size():
    preds = [pred(i * 50, 500) for i in range(20)]  # 40 px holds = 4 in -> 0.1 in/px
    s = scale.estimate_scale(preds, image_height_px=1200)  # 1200 px * 0.1 = 10 ft
    assert s["method"] == "median-hold-size"
    assert s["inches_per_pixel"] == pytest.approx(0.1)


@pytest.mark.parametrize("n", [0, 1, 9])
def test_scale_too_few_holds_falls_back(n):
    s = scale.estimate_scale([pred(i, i) for i in range(n)], image_height_px=1000)
    assert s["method"] == "fallback-wall-height"
    assert s["inches_per_pixel"] == pytest.approx(14 * 12 / 1000)


def test_scale_implausible_wall_falls_back():
    preds = [pred(i * 50, 500, size=4) for i in range(20)]  # tiny holds -> 1 in/px -> 83 ft wall
    assert scale.estimate_scale(preds, 1000)["method"] == "fallback-wall-height"


def test_scale_ignores_low_confidence_and_wrong_class():
    preds = [pred(i, i, conf=0.1) for i in range(20)] + [pred(i, i, class_id=0) for i in range(20)]
    assert scale.estimate_scale(preds, 1000)["holds_used"] == 0


def test_scale_zero_size_boxes_do_not_crash():
    s = scale.estimate_scale([pred(i, i, size=0) for i in range(20)], 1000)
    assert s["method"] == "fallback-wall-height"


# ---------------------------------------------------------------- /detect

def b64_png(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def test_decode_image_accepts_data_url():
    img = wall_with_hold((0, 0, 220))
    assert vision.decode_image("data:image/png;base64," + b64_png(img)).shape == img.shape


@pytest.mark.parametrize("bad", ["not base64!!", base64.b64encode(b"hello").decode(), ""])
def test_decode_image_rejects_garbage(bad):
    with pytest.raises(HTTPException) as e:
        vision.decode_image(bad)
    assert e.value.status_code == 400


def test_to_response_filters_and_orders_bottom_first():
    img = np.zeros((1000, 800, 3), np.uint8)
    preds = [pred(10, 100), pred(20, 900, color="blue"), pred(30, 500, conf=0.2), pred(40, 300, class_id=0)]
    r = vision.to_response(img, preds, min_conf=0.5)
    assert [(h.id, h.y) for h in r.holds] == [(0, 900), (1, 100)]
    assert r.colors == {"blue": 1, "red": 1}
    assert (r.image_width, r.image_height) == (800, 1000)


def test_to_response_with_no_holds():
    r = vision.to_response(np.zeros((1000, 800, 3), np.uint8), [], 0.5)
    assert r.holds == [] and r.scale_method == "fallback-wall-height" and r.inches_per_pixel > 0


def test_detect_endpoint(monkeypatch):
    monkeypatch.setattr(vision, "predict", lambda img: [pred(10, 100), pred(10, 200)])
    r = TestClient(vision.app).post("/detect", json={"image_b64": b64_png(wall_with_hold((0, 0, 220)))})
    assert r.status_code == 200 and len(r.json()["holds"]) == 2


def test_detect_endpoint_roboflow_failure_is_502(monkeypatch):
    def boom(img):
        raise RuntimeError("Roboflow down")
    monkeypatch.setattr(vision, "predict", boom)
    r = TestClient(vision.app).post("/detect", json={"image_b64": b64_png(wall_with_hold((0, 0, 220)))})
    assert r.status_code == 502


# ---------------------------------------------------------------- predict() with a fake Roboflow

@pytest.fixture
def fake_roboflow(monkeypatch):
    """Install a fake inference_sdk so predict() runs without the network."""
    state = {"response": None}

    class HTTPCallErrorError(Exception):
        def __init__(self, status_code=500):
            self.status_code = status_code

    class HTTPClientError(Exception):
        pass

    class Client:
        def __init__(self, **kw):
            pass

        def run_workflow(self, **kw):
            return state["response"]

    sdk = types.ModuleType("inference_sdk")
    sdk.InferenceHTTPClient = Client
    errors = types.ModuleType("inference_sdk.http.errors")
    errors.HTTPCallErrorError, errors.HTTPClientError = HTTPCallErrorError, HTTPClientError
    monkeypatch.setitem(sys.modules, "inference_sdk", sdk)
    monkeypatch.setitem(sys.modules, "inference_sdk.http", types.ModuleType("inference_sdk.http"))
    monkeypatch.setitem(sys.modules, "inference_sdk.http.errors", errors)
    return state


def test_predict_adds_box_and_color(fake_roboflow):
    fake_roboflow["response"] = [{"model_output": {"predictions": [
        {"x": 100, "y": 100, "width": 40, "height": 40, "confidence": 0.9, "class_id": 1}]}}]
    (p,) = object_detection.predict(wall_with_hold((0, 0, 220)))
    assert p["box"] == [80, 80, 120, 120]
    assert p["color"][0] == "red"


@pytest.mark.parametrize("response", [
    [{"model_output": {"predictions": []}}],
    [{"model_output": {"predictions": None}}],
    [{"model_output": None}],
    [{}],
    [],
])
def test_predict_empty_roboflow_response_means_zero_holds(fake_roboflow, response):
    fake_roboflow["response"] = response
    assert object_detection.predict(wall_with_hold((0, 0, 220))) == []
