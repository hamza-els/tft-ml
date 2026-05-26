"""
OCR pipeline for extracting static UI values from TFT frames.
Reads gold, HP, level, stage, and round from fixed pixel regions.

All crop regions assume 1920x1080 source resolution.
If your VODs are a different resolution, update REGIONS in configs/vision_config.yaml.
"""

import re

import cv2
import easyocr
import numpy as np


# Pixel crops for 1920x1080 — (x, y, w, h)
# TODO: Calibrate ALL regions against actual VOD screenshots before trusting these.
REGIONS: dict[str, tuple[int, int, int, int]] = {
    "gold":   (870, 1020, 80,  30),
    "hp":     (20,  60,   60,  25),
    "level":  (45,  990,  30,  20),
    "stage":  (860, 15,   100, 22),
    # Streak: positive = win streak (flames icon), negative = loss streak (ice icon).
    # OCR reads the number only; the sign must be inferred from icon color separately.
    # TODO: calibrate x,y,w,h once real frames are available
    "streak": (820, 1020, 40,  20),
    # XP is displayed as a progress bar, not text, so we read the numeric label
    # "X / Y" shown beside the bar (e.g. "0 / 4"). Both values extracted below.
    # TODO: calibrate x,y,w,h once real frames are available
    "xp":     (75,  995,  60,  18),
}

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def extract_ui_values(frame: np.ndarray) -> dict[str, float | None]:
    """
    Run OCR on a single frame and return parsed UI values.

    Args:
        frame: BGR image array (from cv2.imread or frame_extractor).

    Returns:
        Dict with keys matching REGIONS; values are parsed floats or None.
    """
    reader = _get_reader()
    results: dict[str, float | None] = {}

    for key, (x, y, w, h) in REGIONS.items():
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        detections = reader.readtext(gray, allowlist="0123456789.", detail=0)
        text = detections[0] if detections else ""
        results[key] = _parse_number(text)

    return results


def _parse_number(text: str) -> float | None:
    match = re.search(r"[\d.]+", text)
    return float(match.group()) if match else None


def parse_xp(text: str) -> tuple[int, int] | None:
    """Parse XP text like '0 / 4' or '2/6' into (current_xp, xp_to_level)."""
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
