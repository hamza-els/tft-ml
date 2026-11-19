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
        self.half_width = int(cfg.get("window_half_width", self.width // 2))

    def hexes(self) -> list[tuple[int, int]]:
        """All (row, col) pairs, front row first."""
        return [(r, c) for r in range(len(self.rows)) for c in range(self.cols)]

    def center(self, row: int, col: int) -> tuple[int, int]:
        r = self.rows[row]
        return int(round(r["first_x"] + col * r["pitch"])), int(r["y"])

    def region(self, row: int, col: int) -> tuple[int, int, int, int]:
        cx, cy = self.center(row, col)
        below = int(self.rows[row].get("crop_below", self.below))
        return cx - self.width // 2, cy - self.above, self.width, self.above + below

    def window(self, row: int, col: int, bar=None) -> tuple[int, int, int, int]:
        """
        The unit crop. With the unit's health bar (src/vision/unit_bar.Bar) it
        is anchored to it: centred on the bar, top just above the star badge,
        so every crop has the badge in the same place. Without one, it is
        centred where units on this hex usually lean to (perspective pushes
        them away from the screen centre). Same size either way.
        """
        r = self.rows[row]
        cx, _ = self.center(row, col)
        x, y, w, h = self.region(row, col)
        top = int(r.get("win_top", 0))
        if bar is not None:
            return anchored(bar, self.half_width, h + top)
        mid = cx + r.get("win_shift", 0.0) + r.get("win_lean", 0.0) * (cx - 960)
        return int(round(mid - self.half_width)), y - top, 2 * self.half_width, h + top

    def crop(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        x, y, w, h = self.window(row, col)
        return frame[max(0, y):y + h, max(0, x):x + w]

    def has_health_bar(self, frame: np.ndarray, row: int, col: int, min_pixels: int = 120) -> bool:
        """True if a green health bar is drawn in the top part of the hex's window (left of centre)."""
        x, y, w, h = self.region(row, col)
        return health_bar_pixels(frame[max(0, y):y + 60, x:x + w - 30])[0] >= min_pixels


BAR_WIDTH = 63          # a full health bar, px
BADGE_ABOVE = 12        # the star badge's top is this far above the bar's top
WINDOW_MARGIN = 4       # px of room above the badge


def anchored(bar, half_width: int, height: int) -> tuple[int, int, int, int]:
    """A unit window of fixed size, centred on the health bar, top just above the star badge."""
    mid = bar.left + BAR_WIDTH / 2
    return int(round(mid - half_width)), bar.top - BADGE_ABOVE - WINDOW_MARGIN, 2 * half_width, height


def health_bar_pixels(crop: np.ndarray) -> tuple[int, int]:
    """(green, red) health-bar pixel counts in a crop. Own units have green bars, enemies red."""
    px = crop.astype(np.int16)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    green = int(((g > 150) & (r < 120) & (b < 120)).sum())
    red = int(((r > 170) & (g < 90) & (b < 90)).sum())
    return green, red
