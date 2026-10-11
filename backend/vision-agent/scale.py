import logging
import numpy as np

log = logging.getLogger("scale")

TYPICAL_HOLD_IN = 4.0           # assumed real size of a "median" hold; calibrate on a wall of known height
MIN_CONF = 0.5
TRIM_LOW, TRIM_HIGH = 10, 90    # drop tiny footchips and volumes (percentiles)
WALL_FT_RANGE = (8, 18)         # plausible height of what's in frame
FALLBACK_WALL_FT = 14


def hold_size_px(p: dict) -> float:
    """One 'diameter' per hold: sqrt of box area (handles long thin holds)."""
    return float(np.sqrt(p["width"] * p["height"]))


def estimate_scale(predictions: list[dict], image_height_px: int,
                   hold_class_id: int = 1) -> dict:
    sizes = np.array([
        hold_size_px(p) for p in predictions
        if p.get("class_id") == hold_class_id and p.get("confidence", 0) >= MIN_CONF
    ])
    sizes = sizes[sizes > 0]  # a zero-size box would divide by zero below

    if len(sizes) < 10:
        log.warning("Only %d holds, too few to trust; using fallback wall height", len(sizes))
        return _fallback(image_height_px, len(sizes))

    lo, hi = np.percentile(sizes, [TRIM_LOW, TRIM_HIGH])
    kept = sizes[(sizes >= lo) & (sizes <= hi)]
    median_px = float(np.median(kept))
    inches_per_pixel = TYPICAL_HOLD_IN / median_px
    wall_ft = image_height_px * inches_per_pixel / 12

    if not (WALL_FT_RANGE[0] <= wall_ft <= WALL_FT_RANGE[1]):
        log.warning("Hold-size scale implies a %.1f ft wall; using fallback", wall_ft)
        return _fallback(image_height_px, len(kept))

    return {
        "inches_per_pixel": round(inches_per_pixel, 4),
        "median_hold_px": round(median_px, 1),
        "implied_wall_ft": round(wall_ft, 1),
        "holds_used": int(len(kept)),
        "method": "median-hold-size",
    }


def _fallback(image_height_px: int, n: int) -> dict:
    ipp = FALLBACK_WALL_FT * 12 / image_height_px
    return {
        "inches_per_pixel": round(ipp, 4),
        "median_hold_px": None,
        "implied_wall_ft": FALLBACK_WALL_FT,
        "holds_used": n,
        "method": "fallback-wall-height",
    }