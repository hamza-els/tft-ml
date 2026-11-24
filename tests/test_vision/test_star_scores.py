"""
Batched star-badge scoring (unit_bar.star_scores_many) against OpenCV's
per-bar cv2.matchTemplate, which it replaced (2026-11-24, ~4x faster per bar,
each bar scored once per frame).

Usage:
    python -m pytest tests/test_vision/test_star_scores.py -v
"""

import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

import src.vision.unit_bar as ub
from src.vision.board import BoardHexes
from src.vision.unit_bar import Bar, green_bars, star_scores, star_scores_many

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _cv_scores(frame, bar):
    """The old way: one cv2.matchTemplate per template on the badge zone."""
    zone = ub._zone(frame, bar)
    tmpl = ub.star_templates()
    t = next(iter(tmpl.values()))
    if zone.shape[0] < t.shape[0] or zone.shape[1] < t.shape[1]:
        return {}
    zone = zone.astype(np.float32)
    return {k: float(cv2.matchTemplate(zone, v, cv2.TM_CCOEFF_NORMED).max()) for k, v in tmpl.items()}


@pytest.fixture(scope="module")
def frames():
    out = [cv2.imread(str(FIXTURES / n)) for n in ("frame_010313.jpg", "frame_030059.jpg")]
    assert all(f is not None for f in out)
    # a unit fixture pasted on black: real badges plus flat (zero-variance) windows
    crop = cv2.imread(str(FIXTURES / "unit_3star_3_items.png"))
    black = np.zeros((1080, 1920, 3), np.uint8)
    black[416:416 + crop.shape[0], 1199:1199 + crop.shape[1]] = crop
    return out + [black]


def _bars(frame, seed):
    rng = random.Random(seed)
    bars = [Bar(x, x + 60, y) for x, y in ((rng.randint(-10, 1930), rng.randint(-10, 1090)) for _ in range(300))]
    bars += [Bar(l + 420, r + 420, t + 280) for l, r, t in green_bars(frame[280:640, 420:1520])]
    bars += [Bar(0, 60, 0), Bar(1915, 1975, 1075), Bar(20, 80, 5)]       # off and partly off screen
    return bars


def test_batched_scores_equal_opencv(frames):
    for i, frame in enumerate(frames):
        bars = _bars(frame, i)
        batched = star_scores_many(frame, bars)
        for bar, got in zip(bars, batched):
            want = _cv_scores(frame, bar)
            assert got.keys() == want.keys()
            for k in want:
                assert abs(got[k] - want[k]) < 1e-4, (bar, k, got[k], want[k])


def test_real_badge_scores_high(frames):
    bar = ub.board_bar(frames[2], BoardHexes(), 2, 6)
    assert bar is not None
    assert max(star_scores(frames[2], bar).values()) >= ub._CFG["badge_min_score"]


def test_cache_is_per_frame(frames):
    bar = Bar(1000, 1060, 400)
    a = star_scores(frames[0], bar)
    assert star_scores(frames[0], bar) is a                       # same frame: looked up
    b = star_scores(frames[1], bar)                               # another frame: scored again
    assert b == pytest.approx(_cv_scores(frames[1], bar), abs=1e-4)
    copy = frames[0].copy()                                       # equal pixels, new array: rescored, same values
    assert star_scores(copy, bar) == pytest.approx(a, abs=1e-6)


def test_templates_found_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "_STARS", None)
    monkeypatch.setattr(ub, "_STAR_MAT", None)
    monkeypatch.chdir(tmp_path)
    assert sorted(ub.star_templates()) == [1, 2, 3]


# ── green bars and the unit-bar rule ─────────────────────────────────────────

def test_green_bars_at_equals_green_bars_on_the_slice(frames):
    for frame in frames:
        for x, y, w, h in [(420, 280, 1100, 360), (300, 650, 400, 150), (-20, -10, 300, 200), (1800, 1000, 300, 200)]:
            x0, y0 = max(0, x), max(0, y)
            want = [(l + x0, r + x0, t + y0) for l, r, t in green_bars(frame[y0:y + h, x0:x + w])]
            assert ub.green_bars_at(frame, (x, y, w, h)) == want


def _bar_frame(color, w=63, h=5):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[500:500 + h, 900:900 + w] = color
    return frame, Bar(900, 900 + w, 500)


PURE_GREEN, MINT = (40, 222, 43), (110, 190, 95)        # BGR: unit bar, nameplate bar


@pytest.mark.parametrize("color, w, h, score, want", [
    (PURE_GREEN, 63, 5, 0.40, True),     # clear badge: enough on its own
    (MINT, 63, 12, 0.40, True),
    (PURE_GREEN, 63, 5, 0.20, True),     # weak badge, unit-shaped pure green bar
    (PURE_GREEN, 63, 4, 0.16, True),
    (PURE_GREEN, 63, 5, 0.10, False),    # badge too weak whatever the bar
    (MINT, 63, 5, 0.20, False),          # nameplate colour
    (PURE_GREEN, 63, 12, 0.20, False),   # nameplate height
    (PURE_GREEN, 90, 5, 0.20, False),    # too wide (scenery strips)
    (PURE_GREEN, 40, 5, 0.20, False),    # half hidden: needs a clear badge
])
def test_is_unit_bar(monkeypatch, color, w, h, score, want):
    frame, bar = _bar_frame(color, w, h)
    monkeypatch.setattr(ub, "star_scores", lambda f, b: {1: score, 2: score - 0.1, 3: score - 0.2})
    assert ub.is_unit_bar(frame, bar) is want


def test_real_unit_bars_pass(frames):
    for row, col in [(2, 6)]:
        bar = ub.board_bar(frames[2], BoardHexes(), row, col)
        assert bar is not None and ub.is_unit_bar(frames[2], bar)


def test_tick_split_bar_is_one_bar():
    # a 63 px bar cut by a 1 px tick through every row: one bar from its real left end
    frame, _ = _bar_frame(PURE_GREEN)
    frame[500:505, 923] = (20, 40, 20)
    assert ub.green_bars_at(frame, (850, 480, 200, 40)) == [(900, 963, 500)]


def test_stacked_bars_are_split_into_their_bands():
    # rider's bar (4 rows) over the BFF's (4 rows), touching through a half-filled row
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[407:411, 1267:1330] = PURE_GREEN
    frame[411, 1275:1310] = PURE_GREEN
    frame[412:416, 1272:1334] = PURE_GREEN
    assert sorted(ub.green_bars_at(frame, (1200, 380, 200, 60))) == [(1267, 1330, 407), (1272, 1334, 412)]


def test_solid_tall_blob_is_not_split():
    frame, _ = _bar_frame(MINT, 63, 12)              # a nameplate bar: one solid block, one bar
    assert ub.green_bars_at(frame, (850, 480, 200, 40)) == [(900, 963, 500)]
    frame, _ = _bar_frame(PURE_GREEN, 63, 10)
    assert ub.green_bars_at(frame, (850, 480, 200, 40)) == [(900, 963, 500)]


def test_tight_stack_is_a_rider_pair(frames):
    # a real badged unit bar, and a badge-less green bar 6 px above it (a rider's, tightly stacked)
    from src.vision.unit_reader import rider_bars
    board = BoardHexes()
    bar = ub.board_bar(frames[2], board, 2, 6)
    frame = frames[2].copy()             # a new array: the per-frame caches don't carry over
    frame[bar.top - 6:bar.top - 2, bar.left - 4:bar.left + 59] = PURE_GREEN
    out = rider_bars(frame, board, [("h26", board.window(2, 6, bar), bar)])
    ats = sorted(at for at, _, _ in out)
    assert len(ats) == 2 and ats[1] == ats[0] + "r"
    low = next(b for at, _, b in out if not at.endswith("r"))
    assert (low.left, low.top) == (bar.left, bar.top)                # the badged, lower bar is the BFF's


def test_sparse_first_row_does_not_hide_a_unit_bar(monkeypatch):
    frame, bar = _bar_frame(PURE_GREEN, 63, 5)
    frame[499, 900:910] = PURE_GREEN                     # a few green pixels just above: the blob's top row
    bar = Bar(900, 963, 499)
    monkeypatch.setattr(ub, "star_scores", lambda f, b: {1: 0.25, 2: 0.1, 3: 0.05})
    assert ub.is_unit_bar(frame, bar)


def test_high_stacked_bff_and_rider_share_their_hex():
    # Hamza, 2026-11-24: Veigar riding a Big Furry Friend on h36; the BFF's bar sits ~85 px over the
    # hex's usual bar spot (loose), the rider's ~87 px above that. Before: two loose bars on h16 and h36.
    import cv2
    from src.vision.unit_reader import UnitReader
    path = Path.home() / "tft_frames_temp/ZXj0GIXDaV0/game_07/frame_033700.jpg"
    if not path.exists():
        pytest.skip("full frames of ZXj0GIXDaV0 not on disk")
    units = {u["at"]: u for u in UnitReader().read(cv2.imread(str(path)))}
    assert units["h36"]["name"] == "Big Furry Friend" and "h36r" in units
    assert not units["h36"].get("loose") and "h16" not in units
