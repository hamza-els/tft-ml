"""
The health bar drawn above each of the player's units, and what hangs off it:

    [badge][=====green bar=====]
           [item][item][item]

  badge  star level: bronze chevron = 1 star, silver diamond = 2 stars,
         gold double chevron = 3 stars (templates in assets/templates/stars/).
  items  up to three ~25 px icons in a row just under the bar.

The bar is found in the frame rather than assumed from the hex: its height
depends on how tall the unit is, and the camera's perspective pushes it
sideways the further the unit stands from the screen centre. Horizontally it
is predictable (measured on ~3,700 labelled units, within ~9 px):

    bar_left = cx + x_offset + x_lean * (cx - 960)

where cx is the centre of the hex or bench slot. Everything here is in
1920x1080 frame coordinates for the calibrated layout; settings are in
configs/vision_config.yaml under unit_bar.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg

_CFG = vision_cfg["unit_bar"]
_MODEL_PAD = 4            # the item classifier sees the icon with 4 px around it (as the item crops)


@dataclass
class Bar:
    left: int      # x of the first green pixel
    right: int     # x just past the last green pixel
    top: int       # y of the top of the green fill

    def badge_region(self) -> tuple[int, int, int, int]:
        dx, dy, w, h = _CFG["badge"]
        return self.left + dx, self.top + dy, w, h

    def item_regions(self) -> list[tuple[int, int, int, int]]:
        dx, dy, size = _CFG["item_dx"], _CFG["item_dy"], _CFG["item_size"]
        return [(self.left + dx + i * _CFG["item_pitch"], self.top + dy, size, size)
                for i in range(3)]


def expected_left(cx: float) -> float:
    """Where the bar of a unit standing at screen x = cx starts."""
    return cx + _CFG["x_offset"] + _CFG["x_lean"] * (cx - 960)


_GREEN_LO, _GREEN_HI = (0, 151, 0), (129, 255, 129)     # B < 130, G > 150, R < 130


_BRIDGE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))


def _bars_in_mask(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Bars in a green mask: (left, right, top). Two things are mended first:
    - gaps of 1-2 px along a row are bridged: tanky units' bars carry dark
      tick marks, and a tick through every row split the bar, so the piece
      left over started ~25 px right of the bar and its badge wasn't found;
    - a blob of two bars stacked a few px apart (a rider's bar over the Big
      Furry Friend's, touching through a half-filled row) is split into its
      bands, each band's own left and top (the badge sits by the lower one).
    """
    if _CFG.get("bridge_gaps", True):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _BRIDGE)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i, (x, y, w, h, area) in enumerate(stats[1:n], start=1):
        if w < _CFG["min_width"] or h > 12 or area < 2 * w:
            continue
        if h >= 8 and _CFG.get("split_stacked", True):
            bands = _bands(labels[y:y + h, x:x + w] == i)
            if len(bands) > 1:
                out += [(int(x + l), int(x + r), int(y + t)) for l, r, t in bands if r - l >= _CFG["min_width"]]
                continue
        out.append((int(x), int(x + w), int(y)))
    return out


def _bands(blob: np.ndarray) -> list[tuple[int, int, int]]:
    """Horizontal bands of a blob split at rows less than 80% filled, each at least 3 rows:
    (left, right, top) within the blob. One band: the blob is one bar."""
    fill = blob.sum(1) / blob.shape[1]
    bands, start = [], None
    for k, full in enumerate(list(fill >= 0.8) + [False]):
        if full and start is None:
            start = k
        elif not full and start is not None:
            if k - start >= 3:
                cols = np.flatnonzero(blob[start:k].any(0))
                bands.append((int(cols[0]), int(cols[-1]) + 1, start))
            start = None
    return bands


def green_bars(img: np.ndarray) -> list[tuple[int, int, int]]:
    """Own (green) health bars in an image: (left, right, top) in image coordinates."""
    return _bars_in_mask(cv2.inRange(img, _GREEN_LO, _GREEN_HI))


def green_bars_at(frame: np.ndarray, region: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    """
    green_bars(frame[region]), in frame coordinates, from the frame's green
    mask made once per frame (the same pixels, so the same bars): making it
    per hex and bench slot was most of the bar search's time.
    """
    x, y, w, h = region
    x, y = max(0, x), max(0, y)
    cache = _frame_cache(frame)
    if cache.get("mask") is None:
        cache["mask"] = cv2.inRange(frame, _GREEN_LO, _GREEN_HI)
    return [(l + x, r + x, t + y) for l, r, t in _bars_in_mask(cache["mask"][y:y + h, x:x + w])]


def is_unit_bar(frame: np.ndarray, bar: Bar) -> bool:
    """
    A green bar is an own unit's when a star badge sits on its left: the
    Little Legend's nameplate bar has a level box there instead. A badge
    scoring at least badge_min_score is enough. A weaker one (down to
    weak_badge_score: mostly the dark bronze 1-star badge on dark ground, or
    one half behind another unit) counts when the bar itself is a unit's:
    unit bars are 3-6 px tall, 56-68 px wide at full health (planning) and
    pure green; nameplate bars are 12 px tall or mint (blue and red ~65-130).
    On 600 planning frames (2026-11-24), 340 bars scored 0.15-0.35; the
    unit-shaped ones looked at were units, and the three nameplates of that
    shape (scores 0.09-0.14, all mint) fail on both counts.
    """
    score = max(star_scores(frame, bar).values(), default=0.0)
    if score >= _CFG["badge_min_score"]:
        return True
    if score < _CFG["weak_badge_score"]:
        return False
    lo, hi = _CFG["unit_bar_width"]
    if not lo <= bar.right - bar.left <= hi:
        return False
    strip = frame[max(0, bar.top):bar.top + 13, bar.left:bar.right]
    green = cv2.inRange(strip, _GREEN_LO, _GREEN_HI) > 0
    rows = green.mean(1) > 0.3
    height = int(np.argmin(rows)) if not rows.all() else len(rows)
    hlo, hhi = _CFG["unit_bar_height"]
    if not hlo <= height <= hhi:
        return False
    px = strip[green]
    limit = _CFG["unit_bar_max_blue_red"]
    return bool(len(px)) and float(px[:, 0].mean()) <= limit and float(px[:, 2].mean()) <= limit


def find_bar(frame: np.ndarray, cx: float, top: int) -> Bar | None:
    """
    The health bar of the unit standing at screen x = cx (centre of its hex or
    bench slot), searching from a little above `top` (the top of the hex/slot
    region) downwards. None if there is no own unit there.

    Only bars starting within max_dx of where this position's bar is expected
    count, which rules out the neighbours' bars. A unit's bar always has its
    star badge on the left; the player's Little Legend has a green bar too
    (under its name, with a level box instead of a badge), so a bar without a
    badge is not taken. On 150 frames, 33 of the 35 badge-less bars were the
    Little Legend's (the other two: an effect and a half-hidden bar); every
    bar with a badge that was looked at was a unit's.
    """
    want = expected_left(cx)
    x0, y0 = max(0, int(want) - 40), max(0, top - _CFG["search_above"])
    zone = (x0, y0, int(want) + 110 - x0, top + _CFG["search_below"] - y0)
    near = sorted((abs(l - want), l, r, t) for l, r, t in green_bars_at(frame, zone)
                  if abs(l - want) <= _CFG["max_dx"])
    for _, left, right, top_y in near:
        bar = Bar(left, right, top_y)
        if is_unit_bar(frame, bar):
            return bar
    return None


def board_bar(frame: np.ndarray, board, row: int, col: int) -> Bar | None:
    """find_bar for a board hex (board: src.vision.board.BoardHexes)."""
    x, y, w, h = board.region(row, col)
    return find_bar(frame, board.center(row, col)[0], y)


def bench_bar(frame: np.ndarray, bench, slot: int) -> Bar | None:
    """find_bar for a bench slot (bench: src.vision.bench.BenchSlots)."""
    return find_bar(frame, bench.center_x(slot), bench.region(slot)[1])


def crop(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    return frame[max(0, y):y + h, max(0, x):x + w]


def outline_score(frame: np.ndarray, region: tuple[int, int, int, int]) -> float:
    """
    How much an item slot looks like an icon (0-1). Icons have a dark outline
    about 2 px thick; the score is the share of the left, right and bottom
    edges where that outline is dark, provided the scenery just outside is
    clearly lighter (so a dark patch of arena doesn't count). The top edge is
    skipped: the bar's dark underside runs along it. Best over small shifts.
    Scenery scores ~0, most icons 0.5+.
    """
    x, y, size, _ = region
    sx = _CFG["item_slack_x"]
    pad = 2 + sx
    patch = crop(frame, (x - pad, y - pad, size + 2 * pad, size + 2 * pad))
    if patch.shape[:2] != (size + 2 * pad, size + 2 * pad):
        return 0.0
    g = patch.astype(np.float32).max(axis=2)
    dark = _CFG["outline_dark"]
    best = 0.0
    for dx in range(-sx, sx + 1):
        for dy in (-1, 0, 1):
            x0, y0 = pad + dx, pad + dy
            x1, y1 = x0 + size - 1, y0 + size - 1
            rows, cols = slice(y0 + 3, y1 - 2), slice(x0 + 3, x1 - 2)
            edges = [np.minimum(g[rows, x0], g[rows, x0 + 1]),
                     np.minimum(g[rows, x1], g[rows, x1 - 1]),
                     np.minimum(g[y1, cols], g[y1 - 1, cols])]
            share = min(float((e < dark).mean()) for e in edges)
            outside = np.concatenate([g[rows, x0 - 2], g[y1 + 2, cols]])
            line = np.concatenate(edges)
            if np.percentile(outside, 75) - np.median(line) > _CFG["outline_contrast"]:
                best = max(best, share)
    return best


def _item_patch(icon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The icon without its 2 px border (outline and scenery), and a mask that
    leaves out the bottom-centre patch where stacking items (Guinsoo's
    Rageblade, ...) draw their stack counter.
    """
    m = _CFG["item_margin"]
    patch = icon[m:-m, m:-m]
    h, w = patch.shape[:2]
    mask = np.ones((h, w), bool)
    mask[int(h * 0.62):, int(w * 0.25):int(w * 0.75)] = False
    return patch, mask


def _unit(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def item_vec(icon: np.ndarray) -> np.ndarray:
    """
    Shape of an item icon: each colour channel centred and normalised on its
    own, then joined, as a unit vector for correlation. (Centring all channels
    together let the dark frame every emblem shares dominate: Fae and Sprykin
    Emblem scored 0.81.)
    """
    patch, mask = _item_patch(icon)
    p = patch.astype(np.float32)
    return _unit(np.concatenate([_unit(p[..., c][mask]) for c in range(3)]))


def item_hue(icon: np.ndarray) -> np.ndarray:
    """Colour of an item icon: hue x saturation histogram, as a unit vector."""
    patch, mask = _item_patch(icon)
    hsv = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [18, 4], [0, 180, 0, 256]).reshape(-1)
    return hist / (np.linalg.norm(hist) + 1e-6)


def item_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Shape correlation x colour similarity of two icons, 0-1. Icons that share
    a layout but differ in colour (the emblems) are told apart by the second
    factor. On the footage templates: same-item pairs 0.50-0.75, different
    items at most 0.61 (2 of 2,328 pairs above 0.50).
    """
    return float(item_vec(a) @ item_vec(b)) * float(item_hue(a) @ item_hue(b))


class UnitItemReader:
    """
    Items held by a unit: the icons under its health bar, matched against the
    templates in assets/templates/unit_items/ (cut from footage by
    scripts/harvest_unit_item_icons.py, so they are the same pixels the game
    draws). Template names are item names where known, else IDs (uitem_XXX);
    "Name__2.png" is a second picture of the same item.

    Radiant items share their art with the normal item (only the frame
    differs), so at this size the two can be confused.
    """

    def __init__(self, folder: Path | None = None, use_model: bool = True):
        from src.vision.item_classifier import ItemClassifier
        from src.vision.items import _ROOT
        self.model = ItemClassifier() if use_model and ItemClassifier.available() else None
        folder = folder or _ROOT / _CFG["item_templates"]
        self.names: list[str] = []
        shapes, hues = [], []
        for path in sorted(Path(folder).glob("*.png")):
            img = cv2.imread(str(path))
            if img is not None and img.shape[:2] == (_CFG["item_size"], _CFG["item_size"]):
                self.names.append(path.stem.split("__")[0])   # Name__2.png = another picture of Name
                shapes.append(item_vec(img))
                hues.append(item_hue(img))
        self.matrix = np.stack(shapes) if shapes else None
        self.hue_matrix = np.stack(hues) if hues else None

    def scores(self, icon: np.ndarray) -> np.ndarray:
        """item_similarity of an icon against every template."""
        return (self.matrix @ item_vec(icon)) * (self.hue_matrix @ item_hue(icon))

    def identify(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> tuple[str | None, float]:
        """Best-matching template name and score, allowing the icon to sit a little off."""
        if self.matrix is None:
            return None, 0.0
        x, y, w, h = region
        sx = _CFG["item_slack_x"]
        best: tuple[str | None, float] = (None, 0.0)
        for dx in range(-sx, sx + 1):
            for dy in (-1, 0, 1):
                icon = crop(frame, (x + dx, y + dy, w, h))
                if icon.shape[:2] != (h, w):
                    continue
                scores = self.scores(icon)
                i = int(np.argmax(scores))
                if scores[i] > best[1]:
                    best = (self.names[i], float(scores[i]))
        return best

    def slots(self, frame: np.ndarray, bar: Bar) -> list[dict]:
        """
        Evidence for each of the three slots: outline score, best template and
        score, "item" (clearly an icon) and "maybe" (some sign of one).

        With the item classifier available (src/vision/item_classifier.py), a
        slot it is very sure about (>= item_model_confidence) takes its
        answer; the template matcher decides the rest (items the model was
        never trained on, unclear icons).
        """
        return self.slots_many(frame, [bar])[0]

    def model_answers(self, frame: np.ndarray, bars: list[Bar]) -> list[list[str | None]]:
        """
        The item classifier's sure answers for every slot of every bar, in one
        batch (one GPU call per frame instead of one per unit); None where it
        isn't sure or no model is available.
        """
        answers = [[None] * 3 for _ in bars]
        if self.model is None or not bars:
            return answers
        p = _MODEL_PAD
        crops, where = [], []
        for b, bar in enumerate(bars):
            for s, (x, y, w, h) in enumerate(bar.item_regions()):
                c = crop(frame, (x - p, y - p, w + 2 * p, h + 2 * p))
                if c.shape[:2] == (h + 2 * p, w + 2 * p):
                    crops.append(c)
                    where.append((b, s))
        for (b, s), (cls, conf) in zip(where, self.model.predict(crops)):
            if conf >= _CFG["item_model_confidence"]:
                answers[b][s] = cls
        return answers

    def slots_many(self, frame: np.ndarray, bars: list[Bar]) -> list[list[dict]]:
        """slots() for several units of one frame, with one batched model call."""
        return [self._slots(frame, bar, said) for bar, said in zip(bars, self.model_answers(frame, bars))]

    def read_many(self, frame: np.ndarray, bars: list[Bar]) -> list[list[str | None]]:
        """read() for several units of one frame, with one batched model call."""
        return [[s["name"] for s in sl[:held_count(sl)]] for sl in self.slots_many(frame, bars)]

    def _slots(self, frame: np.ndarray, bar: Bar, model_says: list[str | None]) -> list[dict]:
        regions = bar.item_regions()
        out = []
        for region, said in zip(regions, model_says):
            if said is not None:
                is_item = not said.startswith("_")
                out.append({"outline": None, "name": said if is_item else None, "score": None,
                            "item": is_item, "maybe": is_item, "by": "model"})
                continue
            outline = outline_score(frame, region)
            name, score = self.identify(frame, region)
            named = score >= _CFG["item_sure_score"] or (outline >= _CFG["item_min_outline"]
                                                          and score >= _CFG["item_min_score"])
            # An item is a slot that matches an item template. A strong outline
            # with no match used to count too (an item without a template), but
            # on Hamza's labels that rule found 71 false items (dark scenery
            # framed like an icon, match scores 0.04-0.13) and none of 559 real
            # ones: every item seen has a template now. Outline alone is only
            # "some sign of an icon".
            out.append({"outline": outline, "name": name if named else None, "score": score,
                        "item": named,
                        "maybe": outline >= _CFG["item_maybe_outline"] or score >= _CFG["item_maybe_score"]})
        return out

    def read(self, frame: np.ndarray, bar: Bar) -> list[str | None]:
        """
        One entry per item the unit holds, left to right: the item's template
        name, or None for an icon that is there but matches no template.
        """
        slots = self.slots(frame, bar)
        return [s["name"] for s in slots[:held_count(slots)]]


def held_count(slots: list[dict]) -> int:
    """
    How many items a unit holds, from the evidence per slot (left to right).

    Items fill the slots from the left: a slot can only hold an item if every
    slot before it does. So the count is the run of item slots from the left;
    a slot with only some sign of an icon ("maybe") still counts when the slot
    right after it clearly holds one (a half-hidden first item). Something
    that looks like an item further right, after an empty slot, is scenery.
    """
    count = 0
    for i, s in enumerate(slots):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        if s["item"] or (s["maybe"] and nxt is not None and nxt["item"]):
            count = i + 1
        else:
            break
    return count


_STARS: dict[int, np.ndarray] | None = None
_STAR_MAT: tuple | None = None
_ROOT = Path(__file__).resolve().parents[2]
# Per frame (the last one seen): badge scores by (bar left, bar top), as one bar is checked by its
# hex, by the loose/rider search and for its star level; and the green mask. The frame is kept
# (identity check), so a frame changed in place after scoring would get stale results.
_SCORED: dict = {"frame": None, "scores": {}}


def _frame_cache(frame: np.ndarray) -> dict:
    if _SCORED["frame"] is not frame:
        _SCORED.clear()
        _SCORED.update(frame=frame, scores={})
    return _SCORED


def star_templates() -> dict[int, np.ndarray]:
    """
    assets/templates/stars/star_<n>.png: the median of hundreds of badges cut
    from footage at the same offset from the bar, so the scenery behind them
    averages out and only the badge remains.
    """
    global _STARS
    if _STARS is None:
        folder = Path(_CFG["star_templates"])
        if not folder.is_absolute():
            folder = _ROOT / folder      # not the working directory: elsewhere no badge matched, so no units
        _STARS = {int(p.stem.split("_")[1]): cv2.imread(str(p)).astype(np.float32)
                  for p in sorted(folder.glob("star_*.png"))}
        if not _STARS:
            raise FileNotFoundError(f"no star templates in {folder}")
    return _STARS


def _star_matrix() -> tuple[list[int], np.ndarray, tuple[int, int]]:
    """Star levels, templates minus their per-channel mean and scaled to unit length (one row
    each), and the template size (h, w): the TM_CCOEFF_NORMED template side."""
    global _STAR_MAT
    if _STAR_MAT is None:
        tmpl = star_templates()
        keys = sorted(tmpl)
        shape = tmpl[keys[0]].shape
        rows = []
        for k in keys:
            t = tmpl[k] - tmpl[k].reshape(-1, 3).mean(0)
            rows.append((t / max(float(np.linalg.norm(t)), 1e-9)).ravel())
        _STAR_MAT = keys, np.stack(rows).astype(np.float32), shape[:2]
    return _STAR_MAT


def _zone(frame: np.ndarray, bar: Bar) -> np.ndarray:
    dx, dy, w, h = _CFG["badge"]
    sx, sy = _CFG["badge_slack"]
    return crop(frame, (bar.left + dx - sx, bar.top + dy - sy, w + 2 * sx, h + 2 * sy))


def star_scores_many(frame: np.ndarray, bars: list[Bar]) -> list[dict[int, float]]:
    """
    star_scores for several bars of one frame. Every badge position of every
    bar is one row of a single matrix product with the templates (normalised
    correlation as cv2.TM_CCOEFF_NORMED, the same numbers to float precision):
    one cv2.matchTemplate per template and bar was ~40% of reading a frame's units.
    """
    cache = _frame_cache(frame)["scores"]
    keys, T, (th, tw) = _star_matrix()
    _, _, w, h = _CFG["badge"]
    sx, sy = _CFG["badge_slack"]
    full = (h + 2 * sy, w + 2 * sx)
    todo, zones = [], []
    for bar in bars:
        key = (bar.left, bar.top)
        if key in cache or key in todo:
            continue
        zone = _zone(frame, bar)
        if zone.shape[0] < th or zone.shape[1] < tw:
            cache[key] = {}                          # badge cut off by the screen edge
        elif zone.shape[:2] != full:                 # partly off screen: fewer positions, one at a time
            cache[key] = _star_scores_cv(zone)
        else:
            todo.append(key)
            zones.append(zone)
    if zones:
        z = np.stack(zones).astype(np.float32)                                   # (n, H, W, 3)
        win = np.lib.stride_tricks.sliding_window_view(z, (th, tw), axis=(1, 2))  # (n, py, px, 3, th, tw)
        n, py, px = win.shape[:3]
        flat = win.transpose(0, 1, 2, 4, 5, 3).reshape(n * py * px, th * tw * 3)
        num = (flat @ T.T).astype(np.float64)            # templates are zero-mean: no window mean needed
        # per window: sum and sum of squares from summed-area tables (float64, as OpenCV)
        z64 = z.astype(np.float64)
        def box(a):
            c = np.zeros((n, a.shape[1] + 1, a.shape[2] + 1, 3))
            c[:, 1:, 1:] = a.cumsum(1).cumsum(2)
            return c[:, th:th + py, tw:tw + px] - c[:, :py, tw:tw + px] - c[:, th:th + py, :px] + c[:, :py, :px]
        s1, s2 = box(z64), box(z64 * z64)                # (n, py, px, 3)
        var = (s2.sum(3) - (s1 * s1).sum(3) / (th * tw)).reshape(-1, 1)
        t = np.sqrt(np.maximum(var, 0))
        # as OpenCV: |num| < t -> num / t; up to 1.125 t -> +-1 (rounding); beyond (a flat window) -> 0
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(np.abs(num) < t, num / t, np.where(np.abs(num) < 1.125 * t, np.sign(num), 0.0))
        best = score.reshape(n, py * px, len(keys)).max(1)
        for key, row in zip(todo, best):
            cache[key] = {k: float(v) for k, v in zip(keys, row)}
    return [cache[(b.left, b.top)] for b in bars]


def _star_scores_cv(zone: np.ndarray) -> dict[int, float]:
    zone = zone.astype(np.float32)
    return {k: float(cv2.matchTemplate(zone, t, cv2.TM_CCOEFF_NORMED).max()) for k, t in star_templates().items()}


def star_scores(frame: np.ndarray, bar: Bar) -> dict[int, float]:
    """Best normalised correlation of each star template around the badge position."""
    return star_scores_many(frame, [bar])[0]


def prefetch_star_scores(frame: np.ndarray, region: tuple[int, int, int, int]) -> None:
    """Score every green bar in a region of the frame in one go (later star_scores calls on the
    same frame are then lookups)."""
    star_scores_many(frame, [Bar(l, r, t) for l, r, t in green_bars_at(frame, region)])


def star_level(frame: np.ndarray, bar: Bar) -> int | None:
    """
    1, 2 or 3 from the badge's shape and colour (bronze chevron, silver
    diamond, gold double chevron). None when no template fits clearly, which
    mostly means the badge is hidden behind a panel or the screen edge.
    """
    scores = star_scores(frame, bar)
    if not scores:
        return None
    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    if ranked[0] < _CFG["star_min_score"] or ranked[0] - ranked[1] < _CFG["star_min_margin"]:
        return None
    return best
