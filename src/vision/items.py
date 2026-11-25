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

    def __init__(self, size: int, margin: int, mask_corner: bool = True):
        self.size, self.margin, self.mask_corner = size, margin, mask_corner
        self.names: list[str] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def prepare(self, icon: np.ndarray) -> np.ndarray:
        if icon.shape[:2] != (self.size, self.size):
            icon = cv2.resize(icon, (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = self.margin
        patch = icon[m:-m, m:-m].astype(np.float32)
        h, w = patch.shape[:2]
        if self.mask_corner:
            patch[:h // 2, :w // 2] = 0      # stack-count badge area
        vec = patch.reshape(-1)
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add(self, name: str, icon: np.ndarray):
        self.names.append(name)
        self._rows.append(self.prepare(icon))
        self._matrix = None

    def load(self, folder: Path, skip: set[str] = frozenset()):
        for path in sorted(folder.glob("*.png")):
            if path.stem in skip:
                continue
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
        self.outline = int(cfg["outline"])
        self.icon_size = self.size - 2 * self.outline
        self.slots = int(cfg["slots"])
        self.outline_needed_from = int(cfg.get("outline_needed_from", self.slots))
        self.empty_below = float(cfg["empty_below"])
        self.min_contrast = float(cfg["min_contrast"])
        self.seen_threshold = float(cfg["seen_threshold"])
        self.official_threshold = float(cfg["official_threshold"])
        self.seen_dir = _ROOT / cfg["seen_dir"]

        self.align_search = int(cfg.get("align_search", 6))
        self.edge_min = np.array(cfg.get("edge_min", [40, 0, 50, 20]), np.float32)
        self.channel = cfg.get("edge_channel", "max")
        self.empty_dark = float(cfg.get("empty_dark", 0.5))
        # The mouse cursor's tip (cream outline; 1 = outline pixel), for cursor_score().
        tip = cv2.imread(str(_ROOT / "assets" / "templates" / "hud" / "cursor_tip.png"), cv2.IMREAD_GRAYSCALE)
        self.cursor_tip = None if tip is None else (tip > 127).astype(np.float32)
        self.cursor_threshold = float(cfg.get("cursor_threshold", 0.75))

        # Bundles ("2 Reforgers": same icon as "Reforger") and loot rewards ("10 gold") are no
        # column items: official art of those is not loaded, and names are folded on the way out.
        from src.state.game_data import load_game_data
        self.aliases = load_game_data().item_aliases
        self.seen = _TemplateSet(self.icon_size, int(cfg["match_margin"]))
        self.seen.load(self.seen_dir)
        self.official = _TemplateSet(self.icon_size, int(cfg["match_margin"]))
        self.official.load(_ROOT / vision_cfg["template_matching"]["item_dir"], skip=set(self.aliases))

    def slot_region(self, i: int, dx: int = 0, dy: int = 0) -> tuple[int, int, int, int]:
        """The whole slot, gold outline included (dx, dy: the column's offset in this frame)."""
        return self.x + dx, int(round(self.y + i * self.pitch)) + dy, self.size, self.size

    def icon_region(self, i: int, dx: int = 0, dy: int = 0) -> tuple[int, int, int, int]:
        """The icon inside the slot's outline (what is matched against templates)."""
        x, y, w, h = self.slot_region(i, dx, dy)
        o = self.outline
        return x + o, y + o, w - 2 * o, h - 2 * o

    # ── where the column is, and which slots show a settled item ──────────────
    #
    # Every slot has the same frame: a 1 px dark line, then 2 px of gold, on all four
    # sides. The column sits at the calibrated place on 99% of frames; on the rest the
    # whole column is ~4 px left (it moves as a block), so it is found per frame from
    # those edges. A slot whose own frame is not where it should be shows no settled
    # item: an item being dragged (follows the cursor), a hovered item (drawn larger
    # with a bright frame), a health bar or the column sliding in. Measured on the
    # brightest colour channel (a red frame is faint in grey). Checked on 9,045 filled
    # slots (2026-11-22): 0.4% fail, nearly all such frames (a settled item under the
    # cursor passes); 0.1% more are empty slots with the cursor over them; no item fails
    # systematically (Radiant items 1 of 196 with an earlier, stricter version).

    # Edge geometry for locate(), relative to a slot window's top-left: (rows, column a,
    # column b) for the vertical edges, (row a, row b, columns) for the horizontal ones;
    # contrast = a - b. Count badges cover the top-left corner, so left/top avoid it.
    _LEFT = (slice(22, 40), 1, -1)
    _RIGHT = (slice(6, 40), 43, 44)
    _TOP = (1, -1, slice(22, 40))
    _BOTTOM = (43, 45, slice(6, 40))

    def _strip(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """The strip around the column as one float channel, and the frame row it starts at."""
        m = self.align_search + 2
        y0 = max(0, self.y - m)
        y1 = int(self.y + self.slots * self.pitch + self.size + m)
        strip = frame[y0:y1, :self.x + self.size + m]
        if self.channel == "max":
            return strip.max(axis=2).astype(np.float32), y0
        return cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32), y0

    def _edges(self, g: np.ndarray, x: int, y: int) -> np.ndarray:
        """Outline contrast (left, right, top, bottom) of the slot window at (x, y) of g. Each
        edge is the better of its two halves: a count badge or the cursor covers one half,
        an item that is not in its slot misses the whole edge."""
        med = np.median
        halves = (slice(4, 22), slice(22, 42))
        out = []
        for a, b in ((1, -1), (43, 44)):                       # left, right
            out.append(max(med(g[y + h.start:y + h.stop, x + a] - g[y + h.start:y + h.stop, x + b])
                           for h in halves))
        for a, b in ((1, -1), (43, 45)):                       # top, bottom
            out.append(max(med(g[y + a, x + h.start:x + h.stop] - g[y + b, x + h.start:x + h.stop])
                           for h in halves))
        return np.array(out, np.float32)

    def _filled(self, frame: np.ndarray, i: int, dx: int = 0, dy: int = 0) -> bool:
        x, y, w, h = self.icon_region(i, dx, dy)
        icon = frame[y:y + h, x:x + w]
        if icon.shape[:2] != (h, w) or icon.mean() < self.empty_below:
            return False
        # Mostly black: an empty slot with the cursor (or a dragged item's corner) over it.
        if (icon.max(axis=2) < 30).mean() > self.empty_dark:
            return False
        # A flat patch is scenery showing through a hidden column, not an icon.
        return cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY).std() >= self.min_contrast

    def locate(self, frame: np.ndarray, strip: tuple[np.ndarray, int] | None = None) -> tuple[int, int]:
        """The column's offset (dx, dy) in this frame from the calibrated place: the shift that
        puts the strongest outline edges where the filled slots' edges should be."""
        g, y0 = strip if strip is not None else self._strip(frame)
        # the overlay-prone last slot(s) would pull the fit towards whatever covers them
        slots = [i for i in range(self.outline_needed_from) if self._filled(frame, i)]
        if not slots:
            return 0, 0
        d = np.arange(-self.align_search, self.align_search + 1)
        tops = [(x, y - y0) for x, y, _, _ in map(self.slot_region, slots)]
        score = np.zeros(len(d), np.float32)
        for x, y in tops:
            for rows, a, b in (self._LEFT, self._RIGHT):
                block = g[y + rows.start:y + rows.stop]
                score += (block[:, x + a + d] - block[:, x + b + d]).mean(axis=0)
        dx = int(d[np.argmax(score)])
        score[:] = 0
        for x, y in tops:
            for a, b, cols in (self._TOP, self._BOTTOM):
                block = g[:, x + dx + cols.start:x + dx + cols.stop]
                score += (block[y + a + d] - block[y + b + d]).mean(axis=1)
        return dx, int(d[np.argmax(score)])

    def column(self, frame: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """(icon crop, settled) per filled slot, top to bottom. Empty when no slot shows its
        frame where it should: the column is hidden or covered (scenery is not items)."""
        g, y0 = strip = self._strip(frame)
        dx, dy = self.locate(frame, strip)
        out = []
        for i in range(self.slots):
            if not self._filled(frame, i, dx, dy):
                continue
            x, y, w, h = self.icon_region(i, dx, dy)
            sx, sy = self.slot_region(i, dx, dy)[:2]
            ok = bool((self._edges(g, sx, sy - y0) >= self.edge_min).all())
            if not ok and i >= self.outline_needed_from:
                continue                     # an overlay over the last slot, not an item
            out.append((frame[y:y + h, x:x + w], ok))
        return out if any(ok for _, ok in out) else []

    def cursor_score(self, icon: np.ndarray) -> float:
        """How much the icon shows the mouse cursor's tip (its cream outline; 0-1). The cursor
        often rests on the column for minutes, so templates must not be cut with it on top.
        Cursor templates 0.78-1.00, clean ones at most 0.73 (2026-11-22, 125 templates)."""
        if self.cursor_tip is None:
            return 0.0
        cream = (icon.min(axis=2) > 140).astype(np.float32)
        th, tw = self.cursor_tip.shape
        cream = cv2.copyMakeBorder(cream, 0, th // 2, 0, tw // 2, cv2.BORDER_CONSTANT, value=0)
        return float(cv2.matchTemplate(cream, self.cursor_tip, cv2.TM_CCORR_NORMED).max())

    def slot_icons(self, frame: np.ndarray) -> list[np.ndarray]:
        """Icon crops (inside the outline) of the slots showing a settled item, top to bottom."""
        return [icon for icon, ok in self.column(frame) if ok]

    def identify(self, icon: np.ndarray) -> str | None:
        """Item name for an icon, or None if it matches nothing well enough."""
        name, score = self.seen.best(icon)
        if score < self.seen_threshold:
            name, score = self.official.best(icon)
            if score < self.official_threshold:
                return None
        base, tag = (name.split("~", 1) + [""])[:2]
        if base in self.aliases:                       # a template still named after a bundle
            base = self.aliases[base]
            if base is None:
                return None
        return base + (f"~{tag}" if tag else "")

    def names(self, column: list[tuple[np.ndarray, bool]]) -> list[str | None]:
        """Item names of a column() reading; None = an item that can't be read (no template yet,
        or not settled in its slot: dragged, hovered, covered)."""
        return [self.identify(icon) if ok else None for icon, ok in column]

    def read(self, frame: np.ndarray) -> list[str | None]:
        """Items in the column, top to bottom (None = an item that can't be read)."""
        return self.names(self.column(frame))
