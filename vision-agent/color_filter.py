"""
Classify the color of a climbing hold.

    from color_filter import classify_color
    name, bgr, conf = classify_color(img, hold["box"], hold.get("rle_mask"))

How it works:
  1. Pick the hold's pixels: the mask if there is one, otherwise the box minus
     anything that looks like the wall around it.
  2. Drop chalk/shadow pixels (too gray or too dark), unless most of the hold is
     gray/dark, in which case it's a white, gray or black hold.
  3. Each remaining pixel votes for a color by hue. The winner is the label and
     its share of the votes is the confidence.
Voting (not averaging) keeps a small blue hold on a red wall from coming out purple.
"""

import cv2
import numpy as np

# OpenCV hue runs 0..179. Upper bound (exclusive) -> name. Red wraps around.
HUE_BINS = [
    (8, "red"), (22, "orange"), (35, "yellow"), (85, "green"),
    (130, "blue"), (155, "purple"), (172, "pink"), (180, "red"),
]

MIN_SAT = 60    # below this a pixel is "no color" (white/gray/chalk)
MIN_VAL = 50    # below this a pixel is "black"/shadow
WALL_DIST = 30  # how close (in Lab units) a pixel must be to the wall color to be dropped


def decode_rle(rle: dict) -> np.ndarray:
    """COCO compressed RLE {"size": [h, w], "counts": str} -> (h, w) uint8 mask."""
    h, w = rle["size"]
    s = rle["counts"]
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for n in counts:
        flat[pos:pos + n] = val
        pos += n
        val ^= 1
    return flat.reshape((w, h)).T.copy()


def _hold_pixels(img, box, rle_mask=None):
    """Return the BGR pixels (N, 3) that belong to the hold."""
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w - 1), min(y2, img_h - 1)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), np.uint8)

    crop = img[y1:y2 + 1, x1:x2 + 1]

    if rle_mask is not None:
        mask = decode_rle(rle_mask)
        if mask.shape != (img_h, img_w):  # model ran on a resized image
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        sel = mask[y1:y2 + 1, x1:x2 + 1].astype(bool)
        if sel.sum() >= 5:
            return crop[sel]

    # No mask: sample the wall in a ring around the box, drop pixels that match it.
    pad = max(3, int(0.3 * max(x2 - x1, y2 - y1)))
    ox1, oy1 = max(x1 - pad, 0), max(y1 - pad, 0)
    ox2, oy2 = min(x2 + pad, img_w - 1), min(y2 + pad, img_h - 1)
    ring = np.ones((oy2 - oy1 + 1, ox2 - ox1 + 1), bool)
    ring[y1 - oy1:y2 - oy1 + 1, x1 - ox1:x2 - ox1 + 1] = False
    outer = img[oy1:oy2 + 1, ox1:ox2 + 1]
    if ring.sum() == 0:
        return crop.reshape(-1, 3)

    lab = cv2.cvtColor(np.ascontiguousarray(outer), cv2.COLOR_BGR2LAB).astype(np.float32)
    wall = np.median(lab[ring], axis=0)
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    keep = np.linalg.norm(crop_lab - wall, axis=2) > WALL_DIST
    if keep.sum() < 0.1 * keep.size:  # hold is the same color as the wall; use everything
        return crop.reshape(-1, 3)
    return crop[keep]


def classify_color(img: np.ndarray, box, rle_mask: dict | None = None):
    """
    img:      BGR image (cv2.imread)
    box:      (x1, y1, x2, y2) in pixels
    rle_mask: optional prediction["rle_mask"] from Roboflow
    returns:  (color_name, (b, g, r) median color, confidence 0..1)
    """
    px = _hold_pixels(img, box, rle_mask)
    if len(px) == 0:
        return "unknown", (0, 0, 0), 0.0

    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)

    dark = v < MIN_VAL
    gray = (s < MIN_SAT) & ~dark
    colorful = ~dark & ~gray

    # Mostly uncolored -> white / gray / black hold
    if colorful.mean() < 0.25:
        if dark.mean() > gray.mean():
            name, sel = "black", dark
        else:
            sel = gray
            name = "white" if np.median(v[sel]) > 170 else "gray"
        bgr = tuple(int(c) for c in np.median(px[sel], axis=0))
        return name, bgr, round(float(sel.mean()), 3)

    # Colored hold -> hue vote among colorful pixels
    votes = {name: 0 for _, name in HUE_BINS}
    hc = h[colorful]
    lower = 0
    for upper, name in HUE_BINS:
        votes[name] += int(((hc >= lower) & (hc < upper)).sum())
        lower = upper

    name = max(votes, key=votes.get)
    conf = votes[name] / max(len(hc), 1)
    bgr = tuple(int(c) for c in np.median(px[colorful], axis=0))
    return name, bgr, round(float(conf), 3)
