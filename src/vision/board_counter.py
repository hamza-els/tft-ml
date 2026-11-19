"""
The board counter: "units / team size" drawn large and translucent in the
middle of the arena during planning ("8/8", "8/9").

The game's own numbers: units on the board (trait units such as Elderwood
plants are not counted) and the max team size, including every bonus
(traits, items and augments alike). This is the ground truth the cleaned
board is checked against (src/state/units.py).

The text is drawn over the arena, darker or (with free slots) bright blue,
so it is matched on its edges, which look the same on every arena: the
Sobel edge strength of a strip around the counter, scaled by its 99th
percentile. The slash stays put; the left number is right-aligned to it,
the right number left-aligned, so each side is matched on its own against
templates built from footage (scripts/build_counter_templates.py).

Settings in configs/vision_config.yaml under board_counter.
"""

from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg
from src.vision.board import health_bar_pixels

_ROOT = Path(__file__).resolve().parents[2]
CFG = vision_cfg["board_counter"]
OUT_DIR = _ROOT / CFG["templates"]


def edges(frame: np.ndarray) -> np.ndarray:
    """Edge strength of the counter strip, 0-1 (scaled so the strongest 1% of edges are ~1)."""
    x, y, w, h = CFG["strip"]
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return np.clip(m / (np.percentile(m, 99) + 1e-6), 0, 1.5)


def side(edge: np.ndarray, which: str, two_digits: bool = False, slack: int = 0) -> np.ndarray:
    """One side of the slash: the left number (units) or the right number (team size)."""
    key = f"{which}_{'two' if two_digits else 'one'}"
    x0, x1 = CFG["regions"][key]
    y0, y1 = CFG["rows"]
    h, w = edge.shape
    return edge[max(0, y0 - slack):min(h, y1 + slack), max(0, x0 - slack):min(w, x1 + slack)]


class BoardCounter:
    def __init__(self, folder: Path | None = None):
        folder = folder or OUT_DIR
        self.templates: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
        for p in sorted(Path(folder).glob("*.png")):
            which, n = p.stem.split("_")
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[which][int(n)] = img.astype(np.float32) / 255.0

    def available(self) -> bool:
        return bool(self.templates["right"])

    def scores(self, edge: np.ndarray, which: str) -> dict[int, float]:
        slack = int(CFG["slack"])
        out = {}
        for n, t in self.templates[which].items():
            zone = side(edge, which, two_digits=n >= 10, slack=slack)
            if zone.shape[0] < t.shape[0] or zone.shape[1] < t.shape[1]:
                continue
            out[n] = float(cv2.matchTemplate(zone, t, cv2.TM_CCOEFF_NORMED).max())
        return out

    def _pick(self, scores: dict[int, float]) -> int | None:
        if not scores:
            return None
        ranked = sorted(scores.values(), reverse=True)
        best = max(scores, key=scores.get)
        second = ranked[1] if len(ranked) > 1 else 0.0
        if ranked[0] < CFG["min_score"] or ranked[0] - second < CFG["min_margin"]:
            return None
        return best

    def read(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        """
        (units on board, team size); None for a side that can't be read clearly.

        Nothing is read while enemy (red) health bars cross the counter: in PvE
        rounds the monsters stand right on it during planning, and their bars
        were read as digits ("1/3" came out as "2/3").
        """
        x, y, w, h = CFG["strip"]
        if health_bar_pixels(frame[y:y + h, x:x + w])[1] > CFG["max_red_pixels"]:
            return None, None
        edge = edges(frame)
        return self._pick(self.scores(edge, "left")), self._pick(self.scores(edge, "right"))
