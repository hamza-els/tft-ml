"""
The bench: nine fixed slots along the bottom of the board.

Because the slots never move, a bench unit can be cut out by position alone —
no detector needed. This module only knows the geometry; what stands in a slot
is decided elsewhere (scripts/harvest_bench_units.py labels crops from shop
purchases; a classifier over these crops comes next).
"""

import numpy as np

from src.utils.config import vision_cfg


class BenchSlots:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["bench"]
        self.first_center_x = float(cfg["first_center_x"])
        self.pitch = float(cfg["slot_pitch"])
        self.slots = int(cfg["slots"])
        self.width = int(cfg["crop_width"])
        self.y, self.height = int(cfg["y"]), int(cfg["height"])
        pad = cfg.get("window_pad") or {}
        self.pad_left, self.pad_top = int(pad.get("left", 0)), int(pad.get("top", 0))

    def center_x(self, i: int) -> float:
        return self.first_center_x + i * self.pitch

    def region(self, i: int) -> tuple[int, int, int, int]:
        cx = self.center_x(i)
        return int(round(cx - self.width / 2)), self.y, self.width, self.height

    def window(self, i: int) -> tuple[int, int, int, int]:
        """The unit crop: region() padded left and up so the star badge fits (see window_pad)."""
        x, y, w, h = self.region(i)
        return x - self.pad_left, y - self.pad_top, w + self.pad_left, h + self.pad_top

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
