"""
YOLO-based champion detection for TFT board, bench, and shop regions.

Training pipeline:
  1. Label seed frames in data/processed/labeled/ (Roboflow or LabelImg).
  2. Train a YOLOv8 model on the labeled set.
  3. Use the trained model here to auto-annotate new frames (bootstrap loop).

Model weights live in assets/ or are specified via configs/vision_config.yaml.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    label: str          # Champion name, e.g. "Jinx"
    star_level: int     # 1, 2, or 3
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    region: str         # "board" | "bench" | "shop"


class ChampionDetector:
    def __init__(self, weights_path: str):
        """
        Load a trained YOLOv8 model.

        Args:
            weights_path: Path to .pt weights file (e.g. assets/yolo_tft.pt).
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self._weights_path = weights_path

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image array.
            conf_threshold: Minimum confidence to include a detection.

        Returns:
            List of Detection objects for all champions found.
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                label_raw = self.model.names[int(box.cls)]
                label, star = _parse_label(label_raw)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    label=label,
                    star_level=star,
                    confidence=float(box.conf),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    region=_infer_region(int(y1), frame.shape[0]),
                ))
        return detections


def _parse_label(raw: str) -> tuple[str, int]:
    """
    Expected label format from training: 'ChampionName_1star', '_2star', '_3star'.
    Falls back to star level 1 if not found.
    """
    parts = raw.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("1star", "2star", "3star"):
        return parts[0], int(parts[1][0])
    return raw, 1


def _infer_region(y: int, frame_height: int) -> str:
    """Rough vertical split: top 60% = board, next 20% = bench, bottom 20% = shop."""
    ratio = y / frame_height
    if ratio < 0.60:
        return "board"
    if ratio < 0.80:
        return "bench"
    return "shop"
