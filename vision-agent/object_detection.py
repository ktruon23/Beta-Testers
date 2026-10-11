"""
Vision pipeline for Beta Agents.

    wall photo -> Roboflow workflow -> parsed holds (box, outline, size, color)
               -> holds.json + annotated image

Usage:
    python pipeline.py test.jpg                      # call Roboflow
    python pipeline.py test.jpg --from-json raw.json # replay a saved response (no API call)
    python pipeline.py test.jpg --min-conf 0.4 --out-dir output
"""

import logging
import os
import random
import time
import warnings
from dataclasses import  dataclass, field
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from color_filter import classify_color

warnings.filterwarnings("ignore", message=".*legacy channel.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("vision")

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))

HOLD_CLASS_ID = 1  # class 0 ("-") is a stray dataset label, not a hold


@dataclass
class Hold:
    id: int
    detection_id: str
    confidence: float
    center: tuple          # (x, y) in pixels
    center_norm: tuple     # (x, y) in 0..1, y=0 is the top of the photo
    box: tuple             # (x1, y1, x2, y2) in pixels
    area_px: int           # pixels inside the mask
    color_name: str
    color_bgr: tuple
    polygon: list = field(default_factory=list)  # [[x, y], ...] outline points


def predict(
    image: "np.ndarray | str | Path"
) -> list:

    """
    Returns a list of dictionaries.
    \n
    {
        {'width': 32.0,
        'height': 29.0, 
        'x': 317.0, 
        'y': 107.5, 
        'confidence': 0.9158008098602295, 
        'class_id': 1, 
        'class': 'Climbing Holds and Volumes - v14 2023-09-12 5-44pm', 
        'detection_id': '25bc00d6-3e37-436b-9d62-877be0e967c5', 
        'parent_id': 'image', 
        'rle_mask': {'size': [447, 447], 
        'counts': '__S4m0R=000000000000000000000000000000000000000000K500000000000001N10aea1'}, 
        'box': [301, 93, 333, 122], 
        'color': ('red', (105, 77, 218), 1.0)}
    }
    
    """

    # Accept a decoded BGR image (from the API) or a file path (from the CLI).
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read {image}")

    def detect(img: np.ndarray):

        from inference_sdk import InferenceHTTPClient
        from inference_sdk.http.errors import HTTPCallErrorError, HTTPClientError

        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )

        for attempt in range(1, MAX_ATTEMPTS + 1):

            try:

                return client.run_workflow(
                    workflow_id=WORKFLOW_ID,
                    workspace_name=WORKSPACE_NAME,
                    images={"image": img},
                )[0].get("model_output").get("predictions")
            
            except HTTPCallErrorError as e:

                # 4xx means our request is wrong (bad key, bad workflow id) -> retrying won't help
                if e.status_code and 400 <= e.status_code < 500:
                    raise
                last = e
            except HTTPClientError as e:
                last = e

            if attempt < MAX_ATTEMPTS:
                delay = min(2 ** attempt, 30) + random.random()
                log.warning("Roboflow request failed (attempt %d/%d), retrying in %.1fs: %s",
                            attempt, MAX_ATTEMPTS, delay, last)
                time.sleep(delay)

        raise RuntimeError(f"Roboflow failed after {MAX_ATTEMPTS} attempts") from last

    data = detect(img=img)

    for hold in data:

        x = hold.get("x")
        y = hold.get("y")
        height = hold.get("height")
        width = hold.get("width")

        x1 = int(x - width / 2)
        x2 = int(x + width / 2)
        y1 = int(y - height / 2)
        y2 = int(y + height / 2)

        box = [x1, y1, x2, y2]

        hold['box'] = box
        hold['color'] = classify_color(
            img=img, 
            box=box,
            rle_mask=hold.get('rle_mask'),
        )

    return data

if __name__ == "__main__":

    INPUT_PATH = "test.jpg"

    predictions = predict(INPUT_PATH)

    print(predictions)
