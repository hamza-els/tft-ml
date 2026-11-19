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

A position under a tooltip panel, or whose crop the classifier calls
"_covered", comes back as {"at": ..., "covered": True}: unknown, not empty.
"""

import numpy as np

from src.vision.bench import BenchSlots
from src.vision.board_counter import BoardCounter
from src.vision.tooltip import CFG as TOOLTIP_CFG, covered_share, tooltip_boxes
from src.vision.board import BoardHexes
from src.vision.unit_bar import Bar, UnitItemReader, bench_bar, board_bar, star_level
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
        found = [f for f in occupied(frame, self.board, self.bench) if f[0] not in covered]
        if not found:
            return units
        crops = [frame[max(0, y):y + h, max(0, x):x + w] for _, (x, y, w, h), _ in found]
        probs = self.classifier.probabilities(crops)
        for (at, _, bar), p in zip(found, probs):
            k = int(np.argmax(p))
            name = self.classifier.classes[k]
            if name == NONE:
                continue
            if name == COVERED:
                units.append({"at": at, "covered": True})
                continue
            units.append({
                "at": at,
                "name": name,
                "p": round(float(p[k]), 3),
                "star": star_level(frame, bar),
                "items": self.items.read(frame, bar),
            })
        return units


def occupied(frame: np.ndarray, board: BoardHexes, bench: BenchSlots) -> list[tuple[str, tuple, Bar]]:
    """
    (position id, crop window, health bar) for every hex and bench slot with
    an own unit's health bar over it (src/vision/unit_bar.find_bar: the bar
    lined up with the position, with a star badge on its left).

    This replaced a fixed band at the top of each window: short champions
    draw their bar lower and were missed (a quarter of the units on the
    nearest row), while the band picked up the Little Legend's nameplate,
    green spell effects and the cursor.
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
    return out
