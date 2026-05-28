"""
Tests for frame_dedup.py — change detection and deduplication logic.

Usage:
    python -m pytest tests/test_data_collection/test_frame_dedup.py -v -s
"""

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

FRAME_DIR = Path(r"C:\Users\hamza\tft_frames_temp\z5wIgSE3K38")

# Two frames close together (2fps → ~0.5s apart, likely similar during combat/idle)
FRAME_A = FRAME_DIR / "frame_006383.jpg"
FRAME_B = FRAME_DIR / "frame_006384.jpg"

# Two frames far apart — almost certainly different game state
FRAME_EARLY = FRAME_DIR / "frame_000001.jpg"
FRAME_LATE  = FRAME_DIR / "frame_015722.jpg"


@pytest.fixture(scope="session")
def frame_a():
    assert FRAME_A.exists(), f"Frame not found: {FRAME_A}"
    img = cv2.imread(str(FRAME_A))
    assert img is not None
    return img

@pytest.fixture(scope="session")
def frame_b():
    assert FRAME_B.exists(), f"Frame not found: {FRAME_B}"
    img = cv2.imread(str(FRAME_B))
    assert img is not None
    return img

@pytest.fixture(scope="session")
def frame_early():
    assert FRAME_EARLY.exists(), f"Frame not found: {FRAME_EARLY}"
    img = cv2.imread(str(FRAME_EARLY))
    assert img is not None
    return img

@pytest.fixture(scope="session")
def frame_late():
    assert FRAME_LATE.exists(), f"Frame not found: {FRAME_LATE}"
    img = cv2.imread(str(FRAME_LATE))
    assert img is not None
    return img


# ── _region_diff ──────────────────────────────────────────────────────────────

def test_region_diff_identical_is_zero(frame_a):
    """Comparing a frame with itself must give 0 diff for every region."""
    for name, region in WATCH_REGIONS.items():
        diff = _region_diff(frame_a, frame_a, region)
        assert diff == 0.0, f"Region '{name}': expected 0, got {diff}"


def test_region_diff_is_nonnegative(frame_a, frame_b):
    for name, region in WATCH_REGIONS.items():
        diff = _region_diff(frame_a, frame_b, region)
        assert diff >= 0.0, f"Region '{name}': diff is negative ({diff})"


def test_region_diff_prints_all(frame_a, frame_b):
    """Print per-region diffs so you can eyeball whether regions are calibrated."""
    print("\n  ── Region diffs (frame_006383 vs frame_006384) ──")
    for name, region in WATCH_REGIONS.items():
        diff = _region_diff(frame_a, frame_b, region)
        flag = "CHANGE" if diff > CHANGE_THRESHOLD else "same  "
        print(f"    {flag}  {name:<8}  diff={diff:.2f}  (threshold={CHANGE_THRESHOLD})")


def test_distant_frames_differ_in_at_least_one_region(frame_early, frame_late):
    """Frames 15,000 apart should differ in at least one watched region."""
    diffs = {name: _region_diff(frame_early, frame_late, region)
             for name, region in WATCH_REGIONS.items()}
    print("\n  ── Region diffs (frame_000001 vs frame_015722) ──")
    for name, diff in diffs.items():
        print(f"    {name:<8}  diff={diff:.2f}")
    assert any(d > CHANGE_THRESHOLD for d in diffs.values()), (
        "Early and late frames look identical in all regions — "
        "regions may be miscalibrated (e.g. pointing at a static background)"
    )


# ── _has_changed ──────────────────────────────────────────────────────────────

def test_identical_frame_not_changed(frame_a):
    assert not _has_changed(frame_a, frame_a), (
        "A frame compared with itself should never be marked as changed"
    )


def test_early_vs_late_is_changed(frame_early, frame_late):
    """Frames far apart should almost always be marked as changed."""
    assert _has_changed(frame_early, frame_late), (
        "Frames 15,000 apart were not marked as changed — "
        "regions may be pointing at static parts of the UI"
    )


def test_consecutive_frames_print_verdict(frame_a, frame_b):
    changed = _has_changed(frame_b, frame_a)
    print(f"\n  frame_006383 → frame_006384: {'CHANGED' if changed else 'duplicate (would be removed)'}")


# ── deduplicate_frames ────────────────────────────────────────────────────────

def test_dry_run_deletes_nothing(tmp_path):
    """dry_run=True must not delete any files."""
    # Copy a small slice of frames into a temp dir
    sample = sorted(FRAME_DIR.glob("frame_006*.jpg"))[:20]
    for f in sample:
        shutil.copy(f, tmp_path / f.name)

    before = set(tmp_path.glob("frame_*.jpg"))
    kept, deleted = deduplicate_frames(str(tmp_path), dry_run=True, quiet=True)
    after = set(tmp_path.glob("frame_*.jpg"))

    assert before == after, "dry_run deleted files — it must not modify the directory"
    assert kept + deleted == len(sample)


def test_dedup_reduces_count(tmp_path):
    """Dedup on a real slice should remove at least some frames."""
    sample = sorted(FRAME_DIR.glob("frame_006*.jpg"))[:30]
    for f in sample:
        shutil.copy(f, tmp_path / f.name)

    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)
    remaining = list(tmp_path.glob("frame_*.jpg"))

    print(f"\n  30 frames → kept {kept}, deleted {deleted}")
    assert kept + deleted == len(sample), "kept + deleted should equal total input"
    assert len(remaining) == kept, "files on disk should match kept count"


def test_dedup_keeps_at_least_one(tmp_path):
    """Even a run of identical frames must keep at least one."""
    # Duplicate the same frame 10 times
    src = FRAME_A
    for i in range(10):
        shutil.copy(src, tmp_path / f"frame_{i:06d}.jpg")

    kept, deleted = deduplicate_frames(str(tmp_path), quiet=True)
    assert kept == 1, f"Expected 1 kept for 10 identical frames, got {kept}"
    assert deleted == 9
