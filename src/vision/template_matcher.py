"""
OpenCV template matching for static TFT UI icons (items, augments, traits).
Faster and more reliable than YOLO for fixed-position, non-deforming icons.

Template images live in assets/templates/{items,augments,traits}/.
"""

from pathlib import Path

import cv2
import numpy as np


def load_templates(template_dir: str) -> dict[str, np.ndarray]:
    """Load all PNG templates from a directory, keyed by stem filename."""
    templates = {}
    for path in Path(template_dir).glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            templates[path.stem] = img
    return templates


def match_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.80,
) -> list[tuple[int, int, int, int]]:
    """
    Find all locations of template in frame above the confidence threshold.

    Returns:
        List of (x, y, w, h) bounding boxes in frame coordinates.
    """
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    h, w = template.shape[:2]
    boxes = [(int(x), int(y), w, h) for x, y in zip(*locations[::-1])]
    return _nms(boxes)


def _nms(boxes: list[tuple[int, int, int, int]], overlap: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Simple non-maximum suppression to deduplicate overlapping hits."""
    if not boxes:
        return []
    boxes_np = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=float)
    x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= overlap]
    orig = [boxes[i] for i in keep]
    return orig
