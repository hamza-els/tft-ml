"""
The player's half of the board: 4 rows of 7 hexes at fixed screen positions.

Like the bench, the hexes never move, so a unit can be cut out by position:
one window per hex, tall enough to hold the unit and the health bar above it.
A hex counts as occupied when a health bar is drawn in the top of its window.
"""

import numpy as np

from src.utils.config import vision_cfg


class BoardHexes:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["board"]
        self.rows: list[dict] = cfg["rows"]
        self.cols = int(cfg["cols"])
        self.width = int(cfg["crop_width"])
        self.above, self.below = int(cfg["crop_above"]), int(cfg["crop_below"])

    def hexes(self) -> list[tuple[int, int]]:
        """All (row, col) pairs, front row first."""
        return [(r, c) for r in range(len(self.rows)) for c in range(self.cols)]

    def center(self, row: int, col: int) -> tuple[int, int]:
        r = self.rows[row]
        return int(round(r["first_x"] + col * r["pitch"])), int(r["y"])

    def region(self, row: int, col: int) -> tuple[int, int, int, int]:
        cx, cy = self.center(row, col)
        return cx - self.width // 2, cy - self.above, self.width, self.above + self.below

    def crop(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        x, y, w, h = self.region(row, col)
        return frame[max(0, y):y + h, max(0, x):x + w]


def health_bar_pixels(crop: np.ndarray) -> tuple[int, int]:
    """(green, red) health-bar pixel counts in a crop. Own units have green bars, enemies red."""
    px = crop.astype(np.int16)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    green = int(((g > 150) & (r < 120) & (b < 120)).sum())
    red = int(((r > 170) & (g < 90) & (b < 90)).sum())
    return green, red
