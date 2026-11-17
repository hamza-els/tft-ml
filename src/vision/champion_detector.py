"""
YOLO-based champion detection for the TFT board and bench.

Classes are champion names only — star level is not part of the label.
The shop is read by OCR (ocr_pipeline.read_shop), so detections in the shop
strip are dropped.

Training pipeline:
  1. scripts/sample_for_labeling.py → label seed frames in data/processed/labeled/.
  2. scripts/train_yolo.py → trains and installs the weights.
  3. Use the trained model here to auto-annotate new frames (bootstrap loop).

Model weights live in assets/ or are specified via configs/vision_config.yaml.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    label: str          # Champion name, e.g. "Jinx"
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in frame pixels
    region: str         # "board" | "bench"


class ChampionDetector:
    def __init__(
        self,
        weights_path: str,
        board_max_ratio: float = 0.60,
        bench_max_ratio: float = 0.80,
    ):
        """
        Load a trained YOLOv8 model.

        Args:
            weights_path: Path to .pt weights file (e.g. assets/yolo_tft.pt).
            board_max_ratio: Box centres above this fraction of frame height are board.
            bench_max_ratio: Box centres above this (and below board) are bench;
                anything lower is the shop strip and is ignored.
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")

        self._weights_path = weights_path
        self.board_max_ratio = board_max_ratio
        self.bench_max_ratio = bench_max_ratio

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> list[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image array.
            conf_threshold: Minimum confidence to include a detection.

        Returns:
            List of Detection objects for board and bench champions.
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                region = _infer_region((y1 + y2) / 2, frame.shape[0],
                                       self.board_max_ratio, self.bench_max_ratio)
                if region is None:
                    continue
                detections.append(Detection(
                    label=self.model.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    region=region,
                ))
        return detections


def _infer_region(
    y: float,
    frame_height: int,
    board_max_ratio: float = 0.60,
    bench_max_ratio: float = 0.80,
) -> str | None:
    """Rough vertical split by box centre: board, then bench, then shop (None)."""
    ratio = y / frame_height
    if ratio < board_max_ratio:
        return "board"
    if ratio < bench_max_ratio:
        return "bench"
    return None
