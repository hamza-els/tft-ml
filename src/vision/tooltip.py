"""
Tooltip panels over the board.

Hovering a unit, an item, a trait or a shop card opens a dark rectangular
panel with text that hides whatever is under it: a unit's health bar (so the
hex looks empty) or its body (so the crop shows text). Readings under a
tooltip are not evidence of anything; src/state/units.py treats them as
unknown and fills them from the frames around.

Detection: very dark pixels inside the board area, closed so the text inside
the panel doesn't break it up, then only components that are big and almost
perfectly rectangular (arena shadows and dark scenery are not). Settings in
configs/vision_config.yaml under tooltip.
"""

import cv2
import numpy as np

from src.utils.config import vision_cfg

CFG = vision_cfg["tooltip"]


def tooltip_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of every tooltip panel over the board area, in frame pixels."""
    ax, ay, aw, ah = CFG["area"]
    s = int(CFG["scale"])
    zone = frame[ay:ay + ah, ax:ax + aw]
    small = cv2.resize(zone, (aw // s, ah // s), interpolation=cv2.INTER_AREA)
    dark = (small.max(axis=2) < CFG["dark_below"]).astype(np.uint8)
    k = int(CFG["close_px"]) // s * 2 + 1
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    boxes = []
    min_w, min_h = CFG["min_size"]
    for x, y, w, h, area in stats[1:n]:
        if w * s < min_w or h * s < min_h:
            continue
        if area / float(w * h) < CFG["min_fill"]:
            continue
        boxes.append((ax + x * s, ay + y * s, w * s, h * s))
    return boxes


def covered_share(boxes: list[tuple[int, int, int, int]], region: tuple[int, int, int, int]) -> float:
    """Share of a region (x, y, w, h) that lies under the tooltip boxes."""
    x, y, w, h = region
    if not boxes or w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    for bx, by, bw, bh in boxes:
        x0, y0 = max(0, bx - x), max(0, by - y)
        x1, y1 = min(w, bx + bw - x), min(h, by + bh - y)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return float(mask.mean())
