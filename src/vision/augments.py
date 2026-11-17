"""
Reads the augment bar: the row of up to three icons above the board showing
the augments picked so far.

The bar is a dark strip centred on a fixed x. It is one icon wide after the
first pick and grows to three, staying centred; before the first pick (and in
some scenes) it is not drawn at all. So the number of augments is read from
how wide the dark strip is, and each icon is then cut out and matched against
the icon templates in assets/templates/augments/.

The bar also tells whose board is on screen. On the player's own board it is
on the left. While scouting or fighting away the view is flipped and the bar
shown is the other player's, on the right. A game's augments never change once
picked, so a bar on the right is a sure sign the units on screen are not the
player's.

Templates are collected from footage by scripts/harvest_augment_icons.py, which
saves every new icon as aug_XXX.png. Rename a file to the augment's real name
to have states use that name.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg

_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Augment:
    name: str | None      # template name, None if no template matches
    tier: str             # "silver" | "gold" | "prismatic" (from icon colour)
    icon: np.ndarray      # BGR crop of the icon


class AugmentReader:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["augments"]
        self.center_x = float(cfg["center_x"])
        self.y, self.height = int(cfg["y"]), int(cfg["height"])
        self.away_center_x = float(cfg["away_center_x"])
        self.away_y = int(cfg["away_y"])
        self.slot_width = float(cfg["slot_width"])
        self.max_slots = int(cfg["max_slots"])
        self.dark_below = int(cfg["dark_below"])
        self.match_threshold = float(cfg["match_threshold"])
        self.template_dir = _ROOT / cfg["template_dir"]
        self.templates: dict[str, np.ndarray] = {}
        self.reload_templates()

    def reload_templates(self):
        self.templates = {}
        for path in sorted(self.template_dir.glob("*.png")):
            img = cv2.imread(str(path))
            if img is not None:
                self.templates[path.stem] = img

    def _box(self, center_x: float, y: int) -> tuple[int, int, int, int]:
        w = self.slot_width * self.max_slots
        return int(center_x - w / 2), y, int(round(w)), self.height

    @property
    def region(self) -> tuple[int, int, int, int]:
        """Bounding box of the player's own three-icon bar (x, y, w, h)."""
        return self._box(self.center_x, self.y)

    @property
    def away_region(self) -> tuple[int, int, int, int]:
        """Where another player's bar is drawn while viewing their board."""
        return self._box(self.away_center_x, self.away_y)

    def count(self, frame: np.ndarray) -> int:
        """Number of the player's own augment icons shown (0 if the bar is not on screen)."""
        return self._count(frame, self.region)

    def count_away(self, frame: np.ndarray) -> int:
        """Number of icons in the other player's bar; above 0 means the view is not the player's board."""
        return self._count(frame, self.away_region)

    def _count(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> int:
        x, y, w, h = region
        dark = frame[y:y + h, x:x + w].max(axis=2) < self.dark_below
        # The bar's top and bottom edge rows are solid background; icons only
        # break up the middle. A column belongs to the bar if both edges are dark.
        edge = np.concatenate([dark[:3], dark[-3:]]).mean(axis=0) > 0.8
        for n in range(self.max_slots, 0, -1):
            half = n * self.slot_width / 2
            lo = int(round(w / 2 - half)) + 2
            hi = int(round(w / 2 + half)) - 2
            inside = edge[lo:hi].mean()
            outside = np.concatenate([edge[:max(lo - 4, 0)], edge[hi + 4:]])
            if inside > 0.9 and (outside.size == 0 or outside.mean() < 0.5):
                # A dark scene looks like a bar too; a real one has a bright icon in every slot.
                bright = frame[y:y + h, x:x + w].max(axis=2) > 110
                slots = np.array_split(bright[:, lo - 2:hi + 2], n, axis=1)
                if all(s.mean() > 0.04 for s in slots):
                    return n
        return 0

    def icons(self, frame: np.ndarray) -> list[np.ndarray]:
        """Icon crops, left to right."""
        n = self.count(frame)
        left = self.center_x - n * self.slot_width / 2
        crops = []
        for i in range(n):
            x = int(round(left + i * self.slot_width))
            crops.append(frame[self.y:self.y + self.height, x:x + int(self.slot_width)].copy())
        return crops

    def match(self, icon: np.ndarray) -> tuple[str | None, float]:
        """Best-matching template name and its score (None if below the threshold)."""
        best_name, best = None, 0.0
        for name, tmpl in self.templates.items():
            if tmpl.shape != icon.shape:
                tmpl = cv2.resize(tmpl, (icon.shape[1], icon.shape[0]))
            # Trim 2 px so a 1-2 px shift between frames doesn't hurt the score.
            score = float(cv2.matchTemplate(icon, tmpl[2:-2, 2:-2], cv2.TM_CCOEFF_NORMED).max())
            if score > best:
                best_name, best = name, score
        return (best_name, best) if best >= self.match_threshold else (None, best)

    @staticmethod
    def tier(icon: np.ndarray) -> str:
        """Augment tier from the icon's colour: yellow = gold, violet = prismatic, grey = silver."""
        bright = icon[icon.max(axis=2) > 110].astype(float)
        if len(bright) == 0:
            return "silver"
        b, g, r = bright.mean(axis=0)
        if r - b > 40 and g - b > 25:
            return "gold"
        if b - g > 8:
            return "prismatic"
        return "silver"

    def read(self, frame: np.ndarray) -> list[Augment]:
        return [Augment(self.match(icon)[0], self.tier(icon), icon) for icon in self.icons(frame)]
