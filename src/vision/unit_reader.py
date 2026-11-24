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
gets the nearest free hex; its position is then approximate. A champion
riding the Big Furry Friend has its own bar just above the BFF's
(rider_bars()): it is "h<row><col>r", on the BFF's hex.

A position under a tooltip panel, or whose crop the classifier calls
"_covered", comes back as {"at": ..., "covered": True}: unknown, not empty.
"""

import numpy as np

from src.vision.bench import BenchSlots
from src.vision.board_counter import BoardCounter
from src.vision.tooltip import CFG as TOOLTIP_CFG, covered_share, panel_in_crop, tooltip_boxes
from src.vision.board import BoardHexes
from src.utils.config import vision_cfg
from src.vision.unit_bar import (Bar, UnitItemReader, bench_bar, board_bar, expected_left, green_bars,
                                 prefetch_star_scores,
                                 star_level, star_scores)
from src.vision.unit_classifier import UnitClassifier

NONE = "_none"
COVERED = "_covered"      # label: the crop is hidden behind a tooltip
RIDDEN = "Big Furry Friend"   # the unit a rider ("h<row><col>r") sits on
UNKNOWN = "_unknown"          # label: a unit, but which one can't be told


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
        # a tooltip over the unit's own crop (the bar above it still shows): covered, not read
        panel = [panel_in_crop(c) for c in crops]
        units += [{"at": at, "covered": True} for (at, _, _), p in zip(found, panel) if p]
        found = [f for f, p in zip(found, panel) if not p]
        crops = [c for c, p in zip(crops, panel) if not p]
        if not found:
            return units
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
        # The lower unit of a stacked pair is the BFF (Hamza: bottom bar = BFF): the rider sitting on it
        # often fills its crop (held out, 11 of 18 BFF crops under a Rengar read as Rengar, BFF second),
        # so it is taken as the BFF whenever the BFF is among the model's top 3 for it.
        ridden = self.classifier.classes.index(RIDDEN) if RIDDEN in self.classifier.classes else None
        has_rider = {at[:-1] for at, _, _, _ in keep if at.endswith("r")}
        p_of = {at: p for (at, _, _), p in zip(found, probs)}
        if ridden is not None:
            keep = [(at, bar, RIDDEN, float(p_of[at][ridden]))
                    if at in has_rider and name != RIDDEN and ridden in np.argsort(-p_of[at])[:3]
                    else (at, bar, name, prob) for at, bar, name, prob in keep]
        # A stacked bar is only a rider when the unit under it reads as the BFF; otherwise
        # (a unit dragged over another) it is treated like any loose bar, as before.
        below = {at: name for at, _, name, _ in keep}
        loose_ids |= {id(bar) for at, bar, _, _ in keep if at.endswith("r") and below.get(at[:-1]) != RIDDEN}
        # Hamza (2026-11-23): the BFF always carries a rider. Without a rider bar above it the
        # two bars overlap (or one is hidden) and the crop can't say whose bar it is: a unit, name unknown.
        keep = [(at, bar, UNKNOWN if name == RIDDEN and at + "r" not in below else name, prob)
                for at, bar, name, prob in keep]
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
    # every green bar where any hex or slot looks for one, badge-scored in one batch first
    prefetch_star_scores(frame, _search_box(board, bench))
    out = []
    for r, c in board.hexes():
        bar = board_bar(frame, board, r, c)
        if bar is not None:
            out.append((f"h{r}{c}", board.window(r, c, bar), bar))
    out = rider_bars(frame, board, out)
    for i in range(bench.slots):
        bar = bench_bar(frame, bench, i)
        if bar is not None:
            out.append((f"s{i}", bench.window(i, bar), bar))
    return out, loose_bars(frame, board, [o for o in out if o[0][0] == "h"])


_BAR_CFG = vision_cfg["unit_bar"]


def _search_box(board: BoardHexes, bench: BenchSlots) -> tuple[int, int, int, int]:
    """x, y, w, h around every zone unit_bar.find_bar searches (all hexes and slots) and the loose zone."""
    spots = [(board.center(r, c)[0], board.region(r, c)[1]) for r, c in board.hexes()]
    spots += [(bench.center_x(i), bench.region(i)[1]) for i in range(bench.slots)]
    x0 = min(expected_left(cx) for cx, _ in spots) - 40
    x1 = max(expected_left(cx) for cx, _ in spots) + 110
    y0 = min(top for _, top in spots) - _BAR_CFG["search_above"]
    y1 = max(top for _, top in spots) + _BAR_CFG["search_below"]
    lx, ly, lw, lh = _BAR_CFG["loose"]["zone"]
    x0, y0, x1, y1 = min(x0, lx), min(y0, ly), max(x1, lx + lw), max(y1, ly + lh)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
_LOOSE = _BAR_CFG["loose"]
_RIDER = _BAR_CFG["rider"]


def _badged_bars(frame: np.ndarray) -> list[Bar]:
    """Every green bar in the board area with a star badge on its left (a unit's, not a nameplate)."""
    x0, y0, w, h = _LOOSE["zone"]
    out = []
    for left, right, top in green_bars(frame[y0:y0 + h, x0:x0 + w]):
        bar = Bar(left + x0, right + x0, top + y0)
        if max(star_scores(frame, bar).values(), default=0.0) >= _BAR_CFG["badge_min_score"]:
            out.append(bar)
    return out


def rider_bars(frame: np.ndarray, board: BoardHexes,
               claimed: list[tuple[str, tuple, Bar]]) -> list[tuple[str, tuple, Bar]]:
    """
    The champion riding a Big Furry Friend (Sprykin) has its own bar just
    above the BFF's (config unit_bar.rider), and the hex claims the BFF's.
    Hamza (2026-11-23): top bar = the rider, bottom bar = Big Furry Friend.
    Before this the hex behind took the rider's bar as its own unit (12 of 39
    pairs) or it went loose to some free hex (27).

    The pair confuses the hexes either way: on 39 labelled pairs the BFF's hex
    took the BFF's bar in 14, another hex took the rider's in 12, and neither
    bar was taken in 13. So pairs are found among all badged bars (claimed or
    not), each pair's lower bar gets the hex it fits best among those no
    other unit holds (as loose bars do), and the two become "h<row><col>"
    (the BFF) and "h<row><col>r" (the rider). Returns the entries, hex claims
    on either bar replaced.
    """
    (dy0, dy1), (dx0, dx1) = _RIDER["dy"], _RIDER["dx"]
    hexes = [o for o in claimed if o[0][0] == "h"]
    bars = [b for _, _, b in hexes]
    bars += [b for b in _badged_bars(frame) if not any(_same_bar(b, t) for t in bars)]
    pairs, taken = [], set()
    for low in sorted(bars, key=lambda b: -b.top):                  # lowest on screen first
        if id(low) in taken:
            continue
        ups = [b for b in bars if b is not low and id(b) not in taken
               and dy0 <= b.top - low.top <= dy1 and dx0 <= b.left - low.left <= dx1]
        if len(ups) == 1:
            taken |= {id(low), id(ups[0])}
            pairs.append((low, ups[0]))
    if not pairs:
        return claimed
    keep = [o for o in claimed if id(o[2]) not in taken]
    used = {(int(at[1]), int(at[2])) for at, _, _ in keep if at[0] == "h"}
    out = []
    for low, up in pairs:
        spot = _nearest_free_hex(board, low, used, _LOOSE)
        if spot is None:            # nowhere to put it: leave the hexes' own claims as they were
            keep += [o for o in claimed if o[2] is low or o[2] is up]
            continue
        used.add(spot)
        r, c = spot
        out += [(f"h{r}{c}", board.window(r, c, low), low), (f"h{r}{c}r", board.window(r, c, up), up)]
    return keep + out


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
