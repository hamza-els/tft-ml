"""
Read every unit on the player's own board and bench from one frame.

Only meaningful on own-board planning frames (see src/vision/arena.py): during
combat units walk off their hexes, and away/scouting frames show someone
else's board.

For each board hex and bench slot with a unit's health bar over it (occupied()):
  name   the unit classifier's best guess (src/vision/unit_classifier.py);
         a window it calls "_none" (no unit) is left out
  p      the classifier's confidence in that guess, 0-1
  star   1-3 from the badge left of the health bar, None if hidden
  items  item IDs under the bar, left to right (None = an item we have no
         template for)

Positions: "at" is "h<row><col>" for a board hex (row 0 = front row) or
"s<slot>" for a bench slot (0 = leftmost), the same ids as the unit crops.
A unit whose bar no hex claimed (being dragged, off its hex: loose_bars())
gets the nearest free hex; its position is then approximate.

A position under a tooltip panel, or whose crop the classifier calls
"_covered", comes back as {"at": ..., "covered": True}: unknown, not empty.
"""

import numpy as np

from src.vision.bench import BenchSlots
from src.vision.board_counter import BoardCounter
from src.vision.tooltip import CFG as TOOLTIP_CFG, covered_share, tooltip_boxes
from src.vision.board import BoardHexes
from src.utils.config import vision_cfg
from src.vision.unit_bar import (Bar, UnitItemReader, bench_bar, board_bar, expected_left, green_bars,
                                 star_level, star_scores)
from src.vision.unit_classifier import UnitClassifier

NONE = "_none"
COVERED = "_covered"      # label: the crop is hidden behind a tooltip


class UnitReader:
    def __init__(self, classifier: UnitClassifier | None = None):
        self.board, self.bench = BoardHexes(), BenchSlots()
        self.classifier = classifier or UnitClassifier()
        self.items = UnitItemReader()
        self.counter = BoardCounter()

    def read_counter(self, frame: np.ndarray) -> list[int | None] | None:
        """The game's "units / team size" counter as [units, team size] (None where unreadable)."""
        if not self.counter.available():
            return None
        return list(self.counter.read(frame))

    def covered(self, frame: np.ndarray) -> list[str]:
        """Positions whose window is largely under a tooltip panel (src/vision/tooltip.py)."""
        boxes = tooltip_boxes(frame)
        if not boxes:
            return []
        share = TOOLTIP_CFG["covered_share"]
        out = [f"h{r}{c}" for r, c in self.board.hexes()
               if covered_share(boxes, self.board.window(r, c)) >= share]
        out += [f"s{i}" for i in range(self.bench.slots)
                if covered_share(boxes, self.bench.window(i)) >= share]
        return out

    def read(self, frame: np.ndarray) -> list[dict]:
        """
        Units seen on the board and bench. A position under a tooltip is
        returned as {"at": ..., "covered": True} instead: nothing can be said
        about it (a hidden bar would otherwise read as an empty hex).
        """
        covered = set(self.covered(frame))
        units = [{"at": at, "covered": True} for at in sorted(covered)]
        fixed, loose = occupied_split(frame, self.board, self.bench)
        loose_ids = {id(bar) for _, _, bar in loose}
        found = [f for f in fixed + loose if f[0] not in covered]
        if not found:
            return units
        crops = [frame[max(0, y):y + h, max(0, x):x + w] for _, (x, y, w, h), _ in found]
        probs = self.classifier.probabilities(crops)
        keep = []
        for (at, _, bar), p in zip(found, probs):
            k = int(np.argmax(p))
            name = self.classifier.classes[k]
            if name == NONE:
                continue
            if name == COVERED:
                units.append({"at": at, "covered": True})
                continue
            keep.append((at, bar, name, float(p[k])))
        # Items of every unit in the frame in one batched model call.
        items = self.items.read_many(frame, [bar for _, bar, _, _ in keep])
        for (at, bar, name, prob), held in zip(keep, items):
            units.append({
                "at": at,
                "name": name,
                "p": round(prob, 3),
                "star": star_level(frame, bar),
                "items": held,
                **({"loose": True} if id(bar) in loose_ids else {}),
            })
        return units


def occupied(frame: np.ndarray, board: BoardHexes, bench: BenchSlots) -> list[tuple[str, tuple, Bar]]:
    """All positions with a unit: occupied_split()'s two lists together."""
    fixed, loose = occupied_split(frame, board, bench)
    return fixed + loose


def occupied_split(frame: np.ndarray, board: BoardHexes,
                   bench: BenchSlots) -> tuple[list[tuple[str, tuple, Bar]], list[tuple[str, tuple, Bar]]]:
    """
    (position id, crop window, health bar) for every hex and bench slot with
    an own unit's health bar over it (src/vision/unit_bar.find_bar: the bar
    lined up with the position, with a star badge on its left).

    This replaced a fixed band at the top of each window: short champions
    draw their bar lower and were missed (a quarter of the units on the
    nearest row), while the band picked up the Little Legend's nameplate,
    green spell effects and the cursor.

    Returns (claimed by their own hex or slot, loose_bars()). Loose ones are
    not always board units: a unit dragged from the bench floats over the
    board but is not on it until dropped, so cleaning only keeps a loose
    reading where the board counter says the board is short.
    """
    out = []
    for r, c in board.hexes():
        bar = board_bar(frame, board, r, c)
        if bar is not None:
            out.append((f"h{r}{c}", board.window(r, c, bar), bar))
    for i in range(bench.slots):
        bar = bench_bar(frame, bench, i)
        if bar is not None:
            out.append((f"s{i}", bench.window(i, bar), bar))
    return out, loose_bars(frame, board, [o for o in out if o[0][0] == "h"])


_BAR_CFG = vision_cfg["unit_bar"]
_LOOSE = _BAR_CFG["loose"]


def _same_bar(a: Bar, b: Bar) -> bool:
    return abs(a.left - b.left) <= 3 and abs(a.top - b.top) <= 3


def _nearest_free_hex(board: BoardHexes, bar: Bar, used: set, cfg: dict) -> tuple[int, int] | None:
    """The unclaimed hex whose usual bar position is nearest this bar, within cfg["max_distance"] px."""
    best = None
    for r, c in board.hexes():
        if (r, c) in used:
            continue
        cx, _ = board.center(r, c)
        dx = bar.left - expected_left(cx)
        dy = bar.top - (board.region(r, c)[1] + cfg["bar_top"][r])
        d = float(np.hypot(dx, dy))
        if d <= cfg["max_distance"] and (best is None or d < best[0]):
            best = (d, (r, c))
    return best[1] if best else None


def loose_bars(frame: np.ndarray, board: BoardHexes,
               claimed: list[tuple[str, tuple, Bar]]) -> list[tuple[str, tuple, Bar]]:
    """
    Units on the board whose bar no hex claimed: a unit being dragged (it
    floats between hexes, often right over another unit, which kept its own
    bar), or one standing off its hex. Each hex only takes the bar closest to
    where its own bar should be, so these were lost, while the game's board
    counter still counted them: on 200 frames where the counter showed more
    units than were read, 104 had such a bar.

    Every bar with a star badge in the board area that matches no claimed
    bar is given the nearest hex nobody claimed (its position is approximate)
    and a crop anchored to it, so it is classified on its own.
    """
    cfg = _LOOSE
    x0, y0, w, h = cfg["zone"]
    taken = [bar for _, _, bar in claimed]
    used = {(int(at[1]), int(at[2])) for at, _, _ in claimed if at[0] == "h"}
    out = []
    for left, right, top in green_bars(frame[y0:y0 + h, x0:x0 + w]):
        bar = Bar(left + x0, right + x0, top + y0)
        if any(_same_bar(bar, t) for t in taken):
            continue
        if max(star_scores(frame, bar).values(), default=0.0) < _BAR_CFG["badge_min_score"]:
            continue
        spot = _nearest_free_hex(board, bar, used, cfg)
        if spot is None:
            continue
        used.add(spot)
        taken.append(bar)
        r, c = spot
        out.append((f"h{r}{c}", board.window(r, c, bar), bar))
    return out
