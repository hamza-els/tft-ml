"""
Reads the unequipped-item column on the left edge of the screen.

The column is a fixed stack of square slots. An empty slot is black; a filled
one shows the item's icon. Icons are identified against two template sets:

  seen      assets/templates/items_seen/ — icons cut from real footage by
            scripts/harvest_item_icons.py. Same pixels as the game draws, so
            matches are near-exact. Named after the official item when that
            was recognised, otherwise item_XXX.png (rename to give it a name).
  official  assets/templates/items/ — official art (scripts/fetch_item_icons.py).
            The game often draws a different or re-cropped picture, so only
            strong matches are trusted.

Stacked consumables show a count badge over the icon's top-left corner, so
matching ignores that corner.
"""

from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg

_ROOT = Path(__file__).resolve().parents[2]


class _TemplateSet:
    """Templates as unit vectors, so one matrix product scores an icon against all of them."""

    def __init__(self, size: int, margin: int):
        self.size, self.margin = size, margin
        self.names: list[str] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def prepare(self, icon: np.ndarray) -> np.ndarray:
        if icon.shape[:2] != (self.size, self.size):
            icon = cv2.resize(icon, (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = self.margin
        patch = icon[m:-m, m:-m].astype(np.float32)
        h, w = patch.shape[:2]
        patch[:h // 2, :w // 2] = 0          # stack-count badge area
        vec = patch.reshape(-1)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add(self, name: str, icon: np.ndarray):
        self.names.append(name)
        self._rows.append(self.prepare(icon))
        self._matrix = None

    def load(self, folder: Path):
        for path in sorted(folder.glob("*.png")):
            img = cv2.imread(str(path))
            if img is not None:
                self.add(path.stem, img)

    def best(self, icon: np.ndarray) -> tuple[str | None, float]:
        if not self.names:
            return None, 0.0
        if self._matrix is None:
            self._matrix = np.stack(self._rows)
        scores = self._matrix @ self.prepare(icon)
        i = int(np.argmax(scores))
        return self.names[i], float(scores[i])


class ItemColumnReader:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["item_column"]
        self.x, self.y = int(cfg["x"]), int(cfg["y"])
        self.size = int(cfg["slot_size"])
        self.pitch = float(cfg["slot_pitch"])
        self.slots = int(cfg["slots"])
        self.empty_below = float(cfg["empty_below"])
        self.min_contrast = float(cfg["min_contrast"])
        self.seen_threshold = float(cfg["seen_threshold"])
        self.official_threshold = float(cfg["official_threshold"])
        self.seen_dir = _ROOT / cfg["seen_dir"]

        self.seen = _TemplateSet(self.size, int(cfg["match_margin"]))
        self.seen.load(self.seen_dir)
        self.official = _TemplateSet(self.size, int(cfg["match_margin"]))
        self.official.load(_ROOT / vision_cfg["template_matching"]["item_dir"])

    def slot_region(self, i: int) -> tuple[int, int, int, int]:
        return self.x, int(round(self.y + i * self.pitch)), self.size, self.size

    def slot_icons(self, frame: np.ndarray) -> list[np.ndarray]:
        """Crops of the filled slots, top to bottom."""
        icons = []
        for i in range(self.slots):
            x, y, w, h = self.slot_region(i)
            icon = frame[y:y + h, x:x + w]
            if icon.shape[:2] != (h, w) or icon.mean() < self.empty_below:
                continue
            # A flat patch is scenery showing through a hidden column, not an icon.
            if cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY).std() < self.min_contrast:
                continue
            icons.append(icon)
        return icons

    def identify(self, icon: np.ndarray) -> str | None:
        """Item name for an icon, or None if it matches nothing well enough."""
        name, score = self.seen.best(icon)
        if score >= self.seen_threshold:
            return name
        name, score = self.official.best(icon)
        return name if score >= self.official_threshold else None

    def read(self, frame: np.ndarray) -> list[str | None]:
        """Items in the column, top to bottom (None = an icon with no template yet)."""
        return [self.identify(icon) for icon in self.slot_icons(frame)]
