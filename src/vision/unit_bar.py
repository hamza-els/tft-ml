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


def green_bars(img: np.ndarray) -> list[tuple[int, int, int]]:
    """Own (green) health bars in an image: (left, right, top) in image coordinates."""
    px = img.astype(np.int16)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    mask = ((g > 150) & (r < 130) & (b < 130)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [(int(x), int(x + w), int(y)) for x, y, w, h, area in stats[1:n]
            if w >= _CFG["min_width"] and h <= 12 and area >= 2 * w]


def find_bar(frame: np.ndarray, cx: float, top: int) -> Bar | None:
    """
    The health bar of the unit standing at screen x = cx (centre of its hex or
    bench slot), searching from a little above `top` (the top of the hex/slot
    region) downwards. None if there is no own unit there.

    Only bars starting within max_dx of where this position's bar is expected
    count, which rules out the neighbours' bars.
    """
    want = expected_left(cx)
    x0, y0 = max(0, int(want) - 40), max(0, top - _CFG["search_above"])
    zone = frame[y0:top + _CFG["search_below"], x0:int(want) + 110]
    near = [(abs(l + x0 - want), l + x0, r + x0, t + y0) for l, r, t in green_bars(zone)
            if abs(l + x0 - want) <= _CFG["max_dx"]]
    if not near:
        return None
    _, left, right, top_y = min(near)
    return Bar(left, right, top_y)


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


def item_vec(icon: np.ndarray) -> np.ndarray:
    """
    An item icon as a unit vector for correlation matching. The 2 px border
    is dropped (outline and scenery), and so is the bottom-centre patch where
    stacking items (Guinsoo's Rageblade, ...) draw their stack counter.
    """
    m = _CFG["item_margin"]
    patch = icon[m:-m, m:-m].astype(np.float32).copy()
    h, w = patch.shape[:2]
    patch[int(h * 0.62):, int(w * 0.25):int(w * 0.75)] = 0
    v = patch.reshape(-1)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


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

    def __init__(self, folder: Path | None = None):
        from src.vision.items import _ROOT
        folder = folder or _ROOT / _CFG["item_templates"]
        self.names: list[str] = []
        rows = []
        for path in sorted(Path(folder).glob("*.png")):
            img = cv2.imread(str(path))
            if img is not None and img.shape[:2] == (_CFG["item_size"], _CFG["item_size"]):
                self.names.append(path.stem.split("__")[0])   # Name__2.png = another picture of Name
                rows.append(item_vec(img))
        self.matrix = np.stack(rows) if rows else None

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
                scores = self.matrix @ item_vec(icon)
                i = int(np.argmax(scores))
                if scores[i] > best[1]:
                    best = (self.names[i], float(scores[i]))
        return best

    def read(self, frame: np.ndarray, bar: Bar) -> list[str | None]:
        """
        One entry per item the unit holds, left to right: the item's template
        name, or None for an icon that is there but matches no template.

        Items fill the slots from the left, so a slot before one that clearly
        holds an item must hold one too, even if it was read poorly.
        """
        slots = []                             # (is_item, name) per slot
        for region in bar.item_regions():
            outline = outline_score(frame, region)
            name, score = self.identify(frame, region)
            named = score >= _CFG["item_sure_score"] or (outline >= _CFG["item_min_outline"]
                                                          and score >= _CFG["item_min_score"])
            slots.append((named or outline >= _CFG["item_sure_outline"], name if named else None))
        count = max((i + 1 for i, (is_item, _) in enumerate(slots) if is_item), default=0)
        return [name for _, name in slots[:count]]


_STARS: dict[int, np.ndarray] | None = None


def star_templates() -> dict[int, np.ndarray]:
    """
    assets/templates/stars/star_<n>.png: the median of hundreds of badges cut
    from footage at the same offset from the bar, so the scenery behind them
    averages out and only the badge remains.
    """
    global _STARS
    if _STARS is None:
        folder = Path(_CFG["star_templates"])
        _STARS = {int(p.stem.split("_")[1]): cv2.imread(str(p)).astype(np.float32)
                  for p in sorted(folder.glob("star_*.png"))}
    return _STARS


def star_scores(frame: np.ndarray, bar: Bar) -> dict[int, float]:
    """Best normalised correlation of each star template around the badge position."""
    dx, dy, w, h = _CFG["badge"]
    sx, sy = _CFG["badge_slack"]
    zone = crop(frame, (bar.left + dx - sx, bar.top + dy - sy, w + 2 * sx, h + 2 * sy)).astype(np.float32)
    scores = {}
    for stars, tmpl in star_templates().items():
        if zone.shape[0] < tmpl.shape[0] or zone.shape[1] < tmpl.shape[1]:
            return {}
        scores[stars] = float(cv2.matchTemplate(zone, tmpl, cv2.TM_CCOEFF_NORMED).max())
    return scores


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
