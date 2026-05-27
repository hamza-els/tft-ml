"""
Remove near-duplicate frames from an extracted frame directory.

Compares consecutive frames in the key UI regions (gold, shop, bench).
Keeps a frame only if something meaningful changed since the last kept frame.
This naturally produces:
  - High density during rerolls / buying (lots of change)
  - Low density during combat / idle (nothing changes)
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Regions to compare for change detection (x, y, w, h) at 1920x1080
# Only watch areas that change when a player takes an action.
# Combat visuals are deliberately excluded — we don't want combat density.
WATCH_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "gold":  (870, 1020, 80,  30),
    "shop":  (500, 940,  920, 110),   # full shop row
    "bench": (200, 860,  1100, 70),   # bench units
}

# How different a region must look to count as a meaningful change.
# Mean absolute pixel difference per channel, 0-255.
# Lower = more sensitive (keeps more frames). Higher = more aggressive dedup.
CHANGE_THRESHOLD = 8.0


def _region_diff(a: np.ndarray, b: np.ndarray, region: tuple[int, int, int, int]) -> float:
    x, y, w, h = region
    crop_a = a[y:y + h, x:x + w].astype(float)
    crop_b = b[y:y + h, x:x + w].astype(float)
    return float(np.mean(np.abs(crop_a - crop_b)))


def _has_changed(frame: np.ndarray, reference: np.ndarray) -> bool:
    """Return True if any watched region differs enough from the reference frame."""
    return any(
        _region_diff(frame, reference, region) > CHANGE_THRESHOLD
        for region in WATCH_REGIONS.values()
    )


def deduplicate_frames(
    frame_dir: str,
    threshold: float = CHANGE_THRESHOLD,
    dry_run: bool = False,
    quiet: bool = False,
) -> tuple[int, int]:
    """
    Remove near-duplicate frames from a directory of extracted JPEGs.

    Args:
        frame_dir: Path to directory containing frame_XXXXXX.jpg files.
        threshold: MAD threshold for change detection.
        dry_run: If True, report what would be deleted without deleting.

    Returns:
        (kept, deleted) counts.
    """
    paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    if not paths:
        print(f"No frames found in {frame_dir}")
        return 0, 0

    kept = 0
    deleted = 0
    reference: np.ndarray | None = None

    for path in tqdm(paths, desc="Deduplicating", unit="frame", dynamic_ncols=True, disable=quiet,
                     smoothing=0, mininterval=1.0):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        if reference is None or _has_changed(frame, reference):
            reference = frame
            kept += 1
        else:
            if not dry_run:
                path.unlink()
            deleted += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"  Kept {kept} | {action} {deleted} | "
          f"{deleted / max(1, kept + deleted) * 100:.0f}% removed")
    return kept, deleted
