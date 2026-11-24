"""
Reads the augment bar: the row of icons above the board showing the augments
picked so far (three picks, at 2-1, 3-2 and 4-2, and in some games a fourth).

The bar is a dark strip centred near a fixed x (within AugmentReader.SHIFT px;
it varies between games). It is one icon wide after the first pick and grows,
staying centred; before the first pick (and in some scenes) it is not drawn at
all. The number of icons must be known before cutting: a wrong count cuts
crops that straddle two icons. So every count the stage allows
(expected_counts) is tried at every centre near the calibrated one, and the
one whose dark outline and centred icons fit best wins. Each icon is then cut
out and matched against the icon templates in assets/templates/augments/.

The bar also tells whose board is on screen. On the player's own board it is
on the left. While scouting the bar shown is the other player's, on the right.
In combat both show: the board owner's on the left, the visitor's on the
right (see src/vision/arena.py, home_board_rows).

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
    SHIFT = 20     # px the bar's centre may sit away from the calibrated one (_find)
    BAND = 14      # px of scenery each side of the bar checked for not being bar (_fit)
    MIN_FIT = 0.5  # a settled bar scores 0.75-0.95; below 0.5 it is sliding in or half covered
    GAP = 22       # px: a break in the bar's dark outline this short is an icon touching the edge

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
        self._tm_key, self._tm = None, None      # template matrix for match(), see _template_matrix
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

    def count(self, frame: np.ndarray, allowed: set[int] | None = None) -> int:
        """
        Number of the player's own augment icons shown (0 if the bar is not on screen).
        allowed: the counts possible at this stage (expected_counts); the best fitting
        of those is taken. Without it any count from 1 to max_slots may win.
        """
        return self.bar(frame, allowed)[0]

    def bar(self, frame: np.ndarray, allowed: set[int] | None = None) -> tuple[int, float]:
        """(number of icons, the bar's centre x) of the player's own bar."""
        return self._find(frame, self.center_x, self.y, allowed)

    def count_away(self, frame: np.ndarray) -> int:
        """Number of icons in the other player's bar; above 0 means the view is not the player's board."""
        return self._find(frame, self.away_center_x, self.away_y, None)[0]

    def _find(self, frame: np.ndarray, center_x: float, y: int, allowed: set[int] | None) -> tuple[int, float]:
        """
        The count and centre that fit the bar best, or (0, center_x) when nothing fits.

        The bar is centred near center_x but not exactly: some games sit ~15 px to the
        right (8vvYT3TbJTw), and another player's bar moves with the camera. So every
        centre within SHIFT px is tried with every allowed count, and the one whose
        outline and icons fit best wins (_fit). Knowing the count and centre first is
        what makes each crop hold exactly one icon.
        """
        counts = sorted(a for a in (allowed or range(1, self.max_slots + 1)) if 0 < a <= self.max_slots)
        reach = self.SHIFT + 6 + self.BAND
        w = int(round(self.slot_width * self.max_slots)) + 2 * reach
        x = int(round(center_x - w / 2))
        h = self.height
        if not counts or x < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0, center_x
        box = frame[y:y + h, x:x + w]
        edge = self._edge(box)
        # Icon pixels are looked for in the middle rows (see _fit).
        bright = box[h // 4:3 * h // 4].max(axis=2) > 110
        # About 80 candidates per bar, each summing column ranges: done slice by slice that
        # took ~140 ms a frame. Instead: running sums of the edge, and every possible slot
        # (each start x, each of the two slot widths 37.5 px rounds to) judged in one pass.
        cols = bright.sum(axis=0).astype(np.int64)
        cols_sum = np.concatenate([[0], np.cumsum(cols)])
        cols_x = np.concatenate([[0], np.cumsum(cols * np.arange(w))])
        rows = np.concatenate([np.zeros((bright.shape[0], 1), np.int64), np.cumsum(bright, axis=1)], axis=1)
        slots = {}
        for d in {int(np.floor(self.slot_width)), int(np.ceil(self.slot_width))}:
            a = np.arange(0, w - d + 1)
            total = cols_sum[a + d] - cols_sum[a]
            # an icon: bright pixels on at least 4% of the slot, spread over 40%+ of its middle rows
            filled = (total >= 0.04 * bright.shape[0] * d) & (((rows[:, a + d] - rows[:, a]) >= 2).mean(axis=0) >= 0.4)
            centre = (cols_x[a + d] - cols_x[a]) / np.maximum(total, 1) - a   # bright pixels' mean x in the slot
            slots[d] = (filled.tolist(), (np.abs(centre - (d - 1) / 2) / self.slot_width).tolist())
        sums = {"edge": np.concatenate([[0], np.cumsum(edge)]).tolist(), "slots": slots, "w": w,
                "cols": cols_sum.tolist(), "rows_n": bright.shape[0]}
        best, best_score = (0, center_x), -np.inf
        for shift in range(-self.SHIFT, self.SHIFT + 1, 2):
            c = w / 2 + shift
            for n in counts:
                score, ok = self._fit(sums, c, n)
                score -= 0.005 * abs(shift)          # equal fits: the calibrated centre wins
                if ok and score > best_score:
                    best, best_score = (n, x + c), score
        return best

    def _edge(self, box: np.ndarray) -> np.ndarray:
        """Per column of the box: does it look like part of the bar? The bar's top and bottom
        rows are solid dark background; icons only break up the middle, though some reach the
        top rows (Cu-Mg0jpCSs: an icon's ring covers 3 of the 4 top rows), so 5 of the 8 edge
        rows being dark is enough. The box's very last row is scenery (the bar ends one row
        above it), so it is not used."""
        dark = box.max(axis=2) < self.dark_below
        edge = np.concatenate([dark[0:4], dark[-5:-1]]).mean(axis=0) >= 0.6
        # Big icons touch both edges and leave gaps of up to ~18 px; a short gap with bar
        # on both sides is still bar.
        on = np.flatnonzero(edge)
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= self.GAP:
                edge[a:b] = True
        return edge

    def _fit(self, sums: dict, c: float, n: int) -> tuple[float, bool]:
        """
        How well a bar of n icons centred at column c fits, and whether to accept it.

        Score: the dark outline (inside dark, a BAND of scenery just outside not) minus
        how far each slot's bright pixels sit from the slot's centre (icons are centred in
        their slot; cutting 3 icons as 4 puts halves of icons at slot edges), plus the
        "comb": icon pixels near the slot centres minus those at the borders between slots.
        Between two icons there is a dark gap, so a right cut has its borders on dark
        columns; a cut shifted by half a slot or with the wrong count puts its borders
        through icons. The comb decides when the scenery is as dark as the bar (the outline
        then says nothing, and a half-slot shift still looks "centred": each slot holds two
        icon halves; t8ER_nNdjpk games 2-3, Hamza 2026-11-19). Accepted if
        the inside is dark, every slot holds an icon spread over several middle rows (when
        a dark panel slides over the bar only the icons' tops show, plus the panel's light
        edge: one line across every slot), the icons are centred, and the outline fits,
        unless the scenery around is as dark as the bar (vzUwGQ4GxFQ's forest): then the
        outline says nothing and the icons' centring alone decides.

        sums: from _find: running sum of the edge columns, and per slot width and start x
        whether the slot holds an icon and how far off centre its bright pixels are.
        """
        e, w = sums["edge"], sums["w"]
        half = n * self.slot_width / 2
        lo, hi = int(round(c - half)) + 2, int(round(c + half)) - 2
        inside = (e[hi] - e[lo]) / (hi - lo)
        a1, b1 = max(lo - 6 - self.BAND, 0), max(lo - 6, 0)
        a2, b2 = min(hi + 6, w), min(hi + 6 + self.BAND, w)
        size = (b1 - a1) + (b2 - a2)
        out = float(e[b1] - e[a1] + e[b2] - e[a2]) / size if size else 0.0
        offsets, filled = [], True
        left = c - half
        for i in range(n):
            a, b = int(round(left + i * self.slot_width)), int(round(left + (i + 1) * self.slot_width))
            has_icon, off_centre = sums["slots"][b - a]
            if not has_icon[a]:
                filled = False
                continue
            offsets.append(off_centre[a])
        off = float(np.mean(offsets)) if offsets else 0.5
        comb = self._comb(sums, left, n)
        score = float(inside) - out - 2 * off + comb
        # off above 0.2: icons straddle the slots (the bar is mid-animation or cut wrongly);
        # comb below 0: slot borders cut through icons
        ok = inside > 0.8 and filled and off < 0.2 and (comb > 0 or n == 1) and (score >= self.MIN_FIT or out >= 0.5)
        return score, bool(ok)

    def _comb(self, sums: dict, left: float, n: int) -> float:
        """Share of middle rows with icon pixels around the slot centres (+-5 px) minus that
        at the n-1 borders between slots (+-2 px); 0 for a single slot."""
        if n < 2:
            return 0.0
        cs, w, rn = sums["cols"], sums["w"], sums["rows_n"]

        def mean(a: float, b: float) -> float:
            a, b = max(int(round(a)), 0), min(int(round(b)), w)
            return (cs[b] - cs[a]) / ((b - a) * rn) if b > a else 0.0

        sw = self.slot_width
        centres = sum(mean(left + (i + 0.5) * sw - 5, left + (i + 0.5) * sw + 5) for i in range(n)) / n
        borders = sum(mean(left + i * sw - 2, left + i * sw + 2) for i in range(1, n)) / (n - 1)
        return centres - borders

    def icons(self, frame: np.ndarray, allowed: set[int] | None = None) -> list[np.ndarray]:
        """Icon crops, left to right. allowed: counts possible at this stage (expected_counts)."""
        n, center = self.bar(frame, allowed)
        left = center - n * self.slot_width / 2
        crops = []
        for i in range(n):
            x = int(round(left + i * self.slot_width))
            crops.append(frame[self.y:self.y + self.height, x:x + int(self.slot_width)].copy())
        return crops

    def _template_matrix(self, shape: tuple) -> tuple[np.ndarray, list[str]]:
        """Every template trimmed by 2 px, mean-free per channel and unit length, one row each.
        Rebuilt when templates are added (harvest_augment_icons.py adds them while running)."""
        key = (tuple(self.templates), shape)
        if self._tm_key != key:
            h, w = shape[:2]
            rows = []
            for tmpl in self.templates.values():
                if tmpl.shape != shape:
                    tmpl = cv2.resize(tmpl, (w, h))
                t = tmpl[2:-2, 2:-2].astype(np.float32)
                t = t - t.reshape(-1, 3).mean(axis=0)
                n = float(np.linalg.norm(t))
                rows.append((t / n if n > 0 else t).ravel())
            self._tm = np.stack(rows) if rows else np.zeros((0, (h - 4) * (w - 4) * 3), np.float32)
            self._tm_key = key
        return self._tm, list(self.templates)

    def match(self, icon: np.ndarray) -> tuple[str | None, float]:
        """
        Best-matching template name and its score (None if below the threshold).

        The score is OpenCV's TM_CCOEFF_NORMED of the template trimmed by 2 px (so a 1-2 px
        shift between frames doesn't hurt), best over the 5x5 placements, computed for all
        templates in one matrix product: one cv2.matchTemplate call per template took
        ~2.6 ms on a busy machine, 0.5 s per icon with 180 templates.
        """
        names, scores = self.scores(icon)
        if not names:
            return None, 0.0
        k = int(scores.argmax())
        best = float(scores[k])
        return (names[k], best) if best >= self.match_threshold else (None, best)

    def scores(self, icon: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Every template's score against the icon (see match), in template order."""
        mat, names = self._template_matrix(icon.shape)
        if not names:
            return [], np.zeros(0, np.float32)
        h, w = icon.shape[:2]
        th, tw = h - 4, w - 4
        windows = np.lib.stride_tricks.sliding_window_view(icon.astype(np.float32), (th, tw, 3))[:, :, 0]
        p = windows.reshape(-1, th * tw, 3)
        p = (p - p.mean(axis=1, keepdims=True)).reshape(len(p), -1)
        norms = np.maximum(np.linalg.norm(p, axis=1), 1e-6)
        return names, (mat @ p.T / norms).max(axis=1)

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

    def read(self, frame: np.ndarray, allowed: set[int] | None = None) -> list[Augment]:
        return [Augment(self.match(icon)[0], self.tier(icon), icon) for icon in self.icons(frame, allowed)]


PICK_ROUNDS = [(2, 1), (3, 2), (4, 2)]      # augment choices


MAX_AUGMENTS = 4


def expected_counts(stage: float | None, round_num: float | None) -> set[int] | None:
    """
    How many augments the bar can hold at this stage: at least the picks done
    so far (2-1, 3-2 and 4-2 always happen), at most MAX_AUGMENTS. Extra
    augments are common in this set (seen: 3 at 3-2, 3 at 3-5, 4 at 5-3), so
    the stage only gives the minimum; the bar's outline picks the count that
    fits best within the range (AugmentReader._count_allowed).
    None when the stage is unknown (the outline alone decides then).
    """
    if stage is None or round_num is None:
        return None
    now = (int(stage), int(round_num))
    done = sum(1 for p in PICK_ROUNDS if now > p)
    return set(range(max(done, 1), MAX_AUGMENTS + 1))
