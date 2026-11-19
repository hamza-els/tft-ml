"""
Tests for frame_dedup.py — change detection and deduplication logic.

Frames here are synthetic: flat 1080p fills with specific WATCH_REGIONS
painted in. This keeps the suite independent of extracted footage, which is
deleted between sets, and lets each test assert an exact expected outcome.

A real-footage smoke test is included at the bottom, skipped unless
TFT_TEST_FRAME_DIR points at a folder of extracted frames.

Usage:
    python -m pytest tests/test_data_collection/test_frame_dedup.py -v
    TFT_TEST_FRAME_DIR=~/tft_frames_temp/VOD_ID python -m pytest tests/ -v
"""

import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

from src.data_collection.frame_dedup import (
    CHANGE_THRESHOLD,
    WATCH_REGIONS,
    _has_changed,
    _region_diff,
    deduplicate_frames,
)

FRAME_SHAPE = (1080, 1920, 3)

# A board area clear of every WATCH_REGION. Stands in for combat animation,
# which must never on its own count as a change.
# Guarded by test_combat_region_is_outside_every_watch_region.
COMBAT_REGION = (700, 300, 500, 300)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank(value: int = 0) -> np.ndarray:
    """A 1080p BGR frame filled with a uniform value."""
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _fill(frame: np.ndarray, region: tuple[int, int, int, int], value: int) -> np.ndarray:
    """Copy of frame with region (x, y, w, h) painted a uniform value."""
    x, y, w, h = region
    out = frame.copy()
    out[y:y + h, x:x + w] = value
    return out


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _write(dir_path: Path, frames: list[np.ndarray]) -> None:
    """Write frames as frame_000000.jpg ... at max quality (flat fills stay exact)."""
    for i, frame in enumerate(frames):
        cv2.imwrite(str(dir_path / f"frame_{i:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])


def _gold_series(values: list[int]) -> list[np.ndarray]:
    """Frames identical except for the gold region, set to each value in turn."""
    base = _blank(0)
    return [_fill(base, WATCH_REGIONS["gold"], v) for v in values]


# ── _region_diff ──────────────────────────────────────────────────────────────

def test_region_diff_identical_is_zero():
    frame = _blank(30)
    for name, region in WATCH_REGIONS.items():
        assert _region_diff(frame, frame, region) == 0.0, f"{name} differed from itself"


def test_region_diff_is_symmetric():
    a = _blank(0)
    b = _fill(a, WATCH_REGIONS["gold"], 40)
    region = WATCH_REGIONS["gold"]
    assert _region_diff(a, b, region) == _region_diff(b, a, region)


def test_region_diff_matches_known_magnitude():
    """Filling a whole region with +40 should give a mean absolute diff of 40."""
    a = _blank(0)
    b = _fill(a, WATCH_REGIONS["gold"], 40)
    assert _region_diff(a, b, WATCH_REGIONS["gold"]) == pytest.approx(40.0)


def test_region_diff_ignores_changes_outside_the_region():
    a = _blank(0)
    b = _fill(a, COMBAT_REGION, 255)
    assert _region_diff(a, b, WATCH_REGIONS["gold"]) == 0.0


# ── _has_changed ──────────────────────────────────────────────────────────────

def test_identical_frames_not_changed():
    frame = _blank(30)
    assert not _has_changed(frame, frame)


@pytest.mark.parametrize("name", list(WATCH_REGIONS))
def test_change_in_each_watch_region_is_detected(name):
    """Every configured region must actually drive the decision — no dead regions."""
    ref = _blank(0)
    frame = _fill(ref, WATCH_REGIONS[name], 60)
    assert _has_changed(frame, ref), f"a change in the {name!r} region went undetected"


def test_combat_region_is_outside_every_watch_region():
    """Guard: keeps the combat-exclusion test honest if regions are recalibrated."""
    for name, region in WATCH_REGIONS.items():
        assert not _overlaps(COMBAT_REGION, region), (
            f"COMBAT_REGION now overlaps the {name!r} watch region — "
            "pick a different board area for the exclusion tests"
        )


def test_change_outside_watch_regions_is_ignored():
    """Combat animation alone must not count as a change — that is the whole design."""
    ref = _blank(0)
    frame = _fill(ref, COMBAT_REGION, 255)
    assert not _has_changed(frame, ref)


def test_threshold_boundary():
    ref = _blank(0)
    frame = _fill(ref, WATCH_REGIONS["gold"], 20)   # diff of exactly 20
    assert _has_changed(frame, ref, threshold=10.0)
    assert not _has_changed(frame, ref, threshold=30.0)


def test_threshold_argument_is_respected():
    """Regression: _has_changed once hardcoded CHANGE_THRESHOLD and ignored the arg."""
    ref = _blank(0)
    frame = _fill(ref, WATCH_REGIONS["gold"], int(CHANGE_THRESHOLD * 2))
    assert _has_changed(frame, ref, threshold=CHANGE_THRESHOLD)
    assert not _has_changed(frame, ref, threshold=CHANGE_THRESHOLD * 10)


# ── deduplicate_frames ────────────────────────────────────────────────────────

def test_empty_directory_returns_zeros(tmp_path):
    assert deduplicate_frames(str(tmp_path), quiet=True) == (0, 0)


def test_identical_frames_keep_exactly_one(tmp_path):
    _write(tmp_path, [_blank(0) for _ in range(10)])
    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)
    assert (kept, deleted) == (1, 9)
    assert len(list(tmp_path.glob("frame_*.jpg"))) == 1


def test_distinct_frames_are_all_kept(tmp_path):
    _write(tmp_path, _gold_series([0, 60, 120, 180]))
    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)
    assert (kept, deleted) == (4, 0)


def test_counts_sum_to_total_and_match_disk(tmp_path):
    _write(tmp_path, _gold_series([0, 0, 60, 60, 120]))
    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)
    assert kept + deleted == 5
    assert len(list(tmp_path.glob("frame_*.jpg"))) == kept


def test_dry_run_deletes_nothing_but_reports_real_counts(tmp_path):
    _write(tmp_path, _gold_series([0, 0, 60, 60, 120]))
    before = {p.name for p in tmp_path.glob("frame_*.jpg")}

    dry = deduplicate_frames(str(tmp_path), dry_run=True, quiet=True)
    after = {p.name for p in tmp_path.glob("frame_*.jpg")}
    assert before == after, "dry_run modified the directory"

    wet = deduplicate_frames(str(tmp_path), quiet=True)
    assert dry == wet, "dry_run reported different counts than the real run"


def test_reference_is_last_kept_not_previous_frame(tmp_path):
    """
    Gold drifts by 6 per frame with a threshold of 10. Against the previous
    frame nothing ever changes; against the last *kept* frame every second one
    does. Locks in the accumulate-drift behaviour.
    """
    _write(tmp_path, _gold_series([0, 6, 12, 18, 24]))
    kept, deleted = deduplicate_frames(str(tmp_path), threshold=10.0, quiet=True)
    assert (kept, deleted) == (3, 2), (
        "expected frames 0, 2 and 4 to be kept — dedup may be comparing "
        "against the previous frame instead of the last kept one"
    )


def test_threshold_changes_outcome_end_to_end(tmp_path):
    """Regression: the threshold argument was accepted but never forwarded."""
    _write(tmp_path, _gold_series([0, 20, 40, 60]))

    sensitive = deduplicate_frames(str(tmp_path), threshold=10.0, dry_run=True, quiet=True)
    blunt = deduplicate_frames(str(tmp_path), threshold=99.0, dry_run=True, quiet=True)

    assert sensitive == (4, 0)
    assert blunt == (1, 3)


def test_non_frame_files_are_untouched(tmp_path):
    _write(tmp_path, [_blank(0) for _ in range(3)])
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "thumb.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    deduplicate_frames(str(tmp_path), quiet=True)

    assert (tmp_path / "notes.txt").read_text() == "keep me"
    assert (tmp_path / "thumb.png").exists()


def test_unreadable_frame_is_skipped_without_crashing(tmp_path):
    """A truncated JPEG should not abort a multi-hour dedup run."""
    _write(tmp_path, _gold_series([0, 60]))
    (tmp_path / "frame_000099.jpg").write_bytes(b"not a jpeg")

    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)

    assert kept == 2
    assert (tmp_path / "frame_000099.jpg").exists(), "unreadable frame was deleted"
    assert kept + deleted == 2, "unreadable frames are skipped, not counted"


# ── Optional: real-footage smoke test ─────────────────────────────────────────
# Point TFT_TEST_FRAME_DIR at an extracted VOD folder to exercise dedup against
# genuine gameplay, e.g. ~/tft_frames_temp/VOD_ID

_REAL_DIR = os.environ.get("TFT_TEST_FRAME_DIR", "")

requires_real_frames = pytest.mark.skipif(
    not (_REAL_DIR and Path(_REAL_DIR).expanduser().is_dir()),
    reason="set TFT_TEST_FRAME_DIR to a folder of extracted frames",
)


@requires_real_frames
def test_dedup_reduces_count_on_real_footage(tmp_path):
    sample = sorted(Path(_REAL_DIR).expanduser().glob("frame_*.jpg"))[:200]
    assert sample, f"no frame_*.jpg in {_REAL_DIR}"
    for f in sample:
        shutil.copy(f, tmp_path / f.name)

    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)

    assert kept + deleted == len(sample)
    assert len(list(tmp_path.glob("frame_*.jpg"))) == kept
    assert deleted > 0, "no frames removed from real footage — threshold may be too high"


# ── per-slot comparison and board watching (2026-11-19) ───────────────────────

def test_one_bench_slot_change_is_detected():
    # One slot of nine changes by 40: averaged over the whole strip that is
    # ~4.4 (below the threshold); per slot it is 40.
    base = _blank(0)
    x, y, w, h = WATCH_REGIONS["bench"]
    one_slot = _fill(base, (x, y, w // 9, h), 40)
    assert _region_diff(one_slot, base, WATCH_REGIONS["bench"]) < CHANGE_THRESHOLD
    assert _has_changed(one_slot, base)


def test_board_change_keeps_a_frame_at_most_every_two_seconds(tmp_path, monkeypatch):
    import src.data_collection.frame_dedup as D
    # 12 identical frames (2 fps = 6 s); the board "changes" on every frame.
    _write(tmp_path, [_blank(0)] * 12)
    boards = iter([(i % 2 == 0,) for i in range(100)])
    monkeypatch.setattr(D, "board_signature", lambda frame, board=None: next(boards))
    monkeypatch.setattr(D, "is_planning", lambda frame: True)
    kept, _ = deduplicate_frames(str(tmp_path), quiet=True, watch_board=True, board_min_gap_sec=2.0)
    # first frame + a board keep every 2 s (frames 1, 5, 9) - not one per frame
    assert kept == 4


def test_board_is_not_watched_in_combat(tmp_path, monkeypatch):
    import src.data_collection.frame_dedup as D
    _write(tmp_path, [_blank(0)] * 12)
    boards = iter([(i % 2 == 0,) for i in range(100)])
    monkeypatch.setattr(D, "board_signature", lambda frame, board=None: next(boards))
    monkeypatch.setattr(D, "is_planning", lambda frame: False)
    kept, _ = deduplicate_frames(str(tmp_path), quiet=True, watch_board=True)
    assert kept == 1
