"""
The bench: nine fixed slots along the bottom of the board.

Because the slots never move, a bench unit can be cut out by position alone —
no detector needed. This module only knows the geometry; what stands in a slot
is decided elsewhere (scripts/harvest_bench_units.py labels crops from shop
purchases; a classifier over these crops comes next).
"""

import numpy as np

from src.utils.config import vision_cfg
from src.vision.board import anchored


class BenchSlots:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["bench"]
        self.first_center_x = float(cfg["first_center_x"])
        self.pitch = float(cfg["slot_pitch"])
        self.slots = int(cfg["slots"])
        self.width = int(cfg["crop_width"])
        self.y, self.height = int(cfg["y"]), int(cfg["height"])
        self.win = cfg.get("window") or {}

    def center_x(self, i: int) -> float:
        return self.first_center_x + i * self.pitch

    def region(self, i: int) -> tuple[int, int, int, int]:
        cx = self.center_x(i)
        return int(round(cx - self.width / 2)), self.y, self.width, self.height

    def window(self, i: int, bar=None) -> tuple[int, int, int, int]:
        """
        The unit crop. With the unit's health bar it is anchored to it
        (src/vision/board.anchored); without, centred where units in this slot
        usually lean to. Same size either way.
        """
        cx = self.center_x(i)
        x, y, w, h = self.region(i)
        half = int(self.win.get("half_width", w // 2))
        top = int(self.win.get("pad_top", 0))
        if bar is not None:
            return anchored(bar, half, h + top)
        mid = cx + self.win.get("shift", 0.0) + self.win.get("lean", 0.0) * (cx - 960)
        return int(round(mid - half)), y - top, 2 * half, h + top

    def crop(self, frame: np.ndarray, i: int) -> np.ndarray:
        x, y, w, h = self.region(i)
        return frame[y:y + h, x:x + w]

    def slot_changes(self, before: np.ndarray, after: np.ndarray) -> list[float]:
        """Mean absolute pixel change in each slot between two frames."""
        return [
            float(np.abs(self.crop(after, i).astype(np.int16)
                         - self.crop(before, i).astype(np.int16)).mean())
            for i in range(self.slots)
        ]

    def covered(self, frame: np.ndarray) -> list[bool]:
        """
        True for slots hidden behind a tooltip (hovering a shop card or unit
        opens a near-black panel over part of the bench).
        """
        return [
            bool((self.crop(frame, i).max(axis=2) < 45).mean() > 0.35)
            for i in range(self.slots)
        ]

    def has_health_bar(self, frame: np.ndarray, i: int) -> bool:
        """True if the slot shows the green health bar drawn above a bench unit."""
        top = self.crop(frame, i)[:40].astype(np.int16)
        b, g, r = top[..., 0], top[..., 1], top[..., 2]
        return int(((g > 150) & (r < 120) & (b < 120)).sum()) > 40
