"""
OCR pipeline for extracting static UI values from TFT frames.
Reads gold, HP, level, stage, round, streak, and XP from fixed pixel regions.

Regions are loaded from configs/vision_config.yaml (all assume 1920x1080).
"""

import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
import yaml


def _load_regions() -> dict[str, tuple[int, int, int, int]]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "vision_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {k: tuple(v) for k, v in cfg["ocr"]["regions"].items()}


REGIONS: dict[str, tuple[int, int, int, int]] = _load_regions()

# Per-region OCR allowlists
_ALLOWLIST: dict[str, str] = {
    "stage":  "0123456789-",   # reads "3-2"
    "xp":     "0123456789/",   # reads "2 / 6"
    "default":"0123456789.",
}

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def crop_region(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    return frame[y:y + h, x:x + w]


def extract_ui_values(frame: np.ndarray) -> dict[str, float | None]:
    """
    Run OCR on a single frame and return parsed UI values.

    Keys returned:
        gold, hp, level, stage, round, streak, xp_current, xp_needed
    """
    reader = _get_reader()
    results: dict[str, float | None] = {}

    for key, (x, y, w, h) in REGIONS.items():
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Upscale small crops — EasyOCR accuracy drops on tiny text
        if gray.shape[0] < 40:
            scale = 40 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        allowlist = _ALLOWLIST.get(key, _ALLOWLIST["default"])
        detections = reader.readtext(gray, allowlist=allowlist, detail=0)
        text = detections[0].strip() if detections else ""

        if key == "stage":
            parsed = parse_stage(text)
            results["stage"] = float(parsed[0]) if parsed else None
            results["round"] = float(parsed[1]) if parsed else None
        elif key == "xp":
            parsed = parse_xp(text)
            results["xp_current"] = float(parsed[0]) if parsed else None
            results["xp_needed"]  = float(parsed[1]) if parsed else None
        else:
            results[key] = _parse_number(text)

    return results


def _parse_number(text: str) -> float | None:
    match = re.search(r"[\d.]+", text)
    return float(match.group()) if match else None


def parse_stage(text: str) -> tuple[int, int] | None:
    """
    Parse stage text like '3-2' into (stage, round).
    Also handles '3.2' in case OCR reads the dash as a dot.
    """
    match = re.search(r"(\d+)[^0-9](\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_xp(text: str) -> tuple[int, int] | None:
    """Parse XP text like '0 / 4' or '2/6' into (current_xp, xp_to_level)."""
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
