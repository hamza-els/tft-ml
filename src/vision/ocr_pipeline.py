"""
OCR pipeline for extracting static UI values from TFT frames.
Reads gold, HP, level, stage, and round from fixed pixel regions.

All crop regions assume 1920x1080 source resolution.
If your VODs are a different resolution, update REGIONS in configs/vision_config.yaml.
"""

import re

import cv2
import numpy as np
import pytesseract


# Pixel crops for 1920x1080 — (x, y, w, h)
# TODO: Calibrate against actual VOD screenshots before trusting these.
REGIONS: dict[str, tuple[int, int, int, int]] = {
    "gold":   (870, 1020, 80, 30),
    "hp":     (20,  60,   60, 25),
    "level":  (45,  990,  30, 20),
    "stage":  (860, 15,   100, 22),
}

TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789-."


def extract_ui_values(frame: np.ndarray) -> dict[str, float | str | None]:
    """
    Run OCR on a single frame and return parsed UI values.

    Args:
        frame: BGR image array (from cv2.imread or frame_extractor).

    Returns:
        Dict with keys matching REGIONS; values are parsed floats or None.
    """
    results: dict[str, float | str | None] = {}
    for key, (x, y, w, h) in REGIONS.items():
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config=TESSERACT_CONFIG).strip()
        results[key] = _parse_number(text)
    return results


def _parse_number(text: str) -> float | None:
    match = re.search(r"[\d.]+", text)
    return float(match.group()) if match else None
