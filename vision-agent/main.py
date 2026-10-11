"""
VISION AGENT  -- no LLM
Photo -> list of holds (Roboflow instance segmentation) + color + scale estimate.

Run from the repo root:
    python -m uvicorn main:app --app-dir vision-agent --port 8001

The Orchestrator calls POST /detect. Response matches shared.schemas.DetectResponse,
so the Planner/Critic/Coach never need to know which detector ran.
"""

import base64
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

# Make the repo root importable so `shared` works however this file is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from object_detection import HOLD_CLASS_ID, predict  # noqa: E402
from scale import estimate_scale  # noqa: E402
from shared.schemas import DetectRequest, DetectResponse, Hold  # noqa: E402

warnings.filterwarnings("ignore", message=".*legacy channel.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("vision")

app = FastAPI(title="Vision Agent")


def decode_image(image_b64: str) -> np.ndarray:
    if "," in image_b64[:100]:  # strip a data-URL prefix like "data:image/jpeg;base64,"
        image_b64 = image_b64.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    except ValueError as e:
        raise HTTPException(400, f"image_b64 is not valid base64: {e}") from e
    if buf.size == 0:  # cv2.imdecode raises on an empty buffer
        raise HTTPException(400, "image_b64 is empty.")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the image (expected PNG or JPEG).")
    return img


def to_response(img: np.ndarray, predictions: list[dict], min_conf: float) -> DetectResponse:
    preds = [
        p for p in predictions
        if p.get("class_id") == HOLD_CLASS_ID and p.get("confidence", 0) >= min_conf
    ]
    preds.sort(key=lambda p: -p["y"])  # id 0 = lowest hold on the wall

    holds = [
        Hold(
            id=i,
            x=round(p["x"], 1),
            y=round(p["y"], 1),
            type="unknown",
            color=p["color"][0] if p.get("color") else None,  # (name, bgr, conf) -> name
        )
        for i, p in enumerate(preds)
    ]

    scale = estimate_scale(preds, img.shape[0])
    colors = Counter(h.color for h in holds if h.color)

    return DetectResponse(
        holds=holds,
        image_width=img.shape[1],
        image_height=img.shape[0],
        method="roboflow",
        inches_per_pixel=scale["inches_per_pixel"],
        scale_method=scale["method"],
        colors=dict(colors.most_common()),
    )


@app.get("/")
def health():
    return {"agent": "vision", "status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    img = decode_image(req.image_b64)
    try:
        predictions = predict(img)
    except Exception as e:
        log.exception("Roboflow detection failed")
        raise HTTPException(502, f"Hold detection failed: {e}") from e

    resp = to_response(img, predictions, req.min_conf)
    log.info("Detected %d holds, scale %.3f in/px (%s)",
             len(resp.holds), resp.inches_per_pixel, resp.scale_method)
    return resp


if __name__ == "__main__":
    # Quick local check without the API: python main.py [image]
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read {path}")
    print(to_response(image, predict(image), 0.5).model_dump_json(indent=2))
