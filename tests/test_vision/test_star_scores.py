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
