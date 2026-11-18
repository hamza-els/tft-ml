"""
Remove near-duplicate frames from an extracted frame directory.

Compares each frame with the last kept frame in the key UI regions (gold,
shop, bench, items, stage). Keeps a frame only if something meaningful changed.
This naturally produces:
  - High density during rerolls / buying (lots of change)
  - Low density during combat / idle (nothing changes)

Two optional refinements (used by scripts/dedup_frames.py and the pipeline):
  - HUD-aware: while the bottom bar is hidden (carousel, augment choice, shop
    collapsed) the shop/bench regions show the moving scene, not player
    actions, so only a stage change or the bar reappearing counts as a change.
  - Keep-alive: keep at least one frame every `max_gap_sec`, so slow changes
    that stay under the threshold (e.g. one unit moved) are never lost.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Regions to compare for change detection (x, y, w, h) at 1920x1080.
# Calibrated for the same HUD layout as configs/vision_config.yaml.
# Only watch areas that change when a player takes an action.
# Combat visuals are deliberately excluded — we don't want combat density.
WATCH_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "gold":  (1028, 878, 56, 32),
    "shop":  (552, 927, 1010, 150),   # full shop row
    "bench": (379, 697, 1031, 128),   # bench units
    "items": (8, 259, 50, 484),       # first 9 item slots with outlines; the 10th is rarely used and overlays cover it
    "stage": (762, 2, 100, 34),       # stage text, both bar layouts
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


# Regions on the bottom bar, which can sit further left in some VODs (see hud.py).
BOTTOM_BAR = {"gold", "shop"}


def _has_changed(
    frame: np.ndarray,
    reference: np.ndarray,
    threshold: float = CHANGE_THRESHOLD,
    regions: list[str] | None = None,
    bottom_dx: int = 0,
) -> bool:
    """Return True if any watched region differs enough from the reference frame."""
    names = regions if regions is not None else list(WATCH_REGIONS)
    for name in names:
        x, y, w, h = WATCH_REGIONS[name]
        if name in BOTTOM_BAR:
            x += bottom_dx
        if _region_diff(frame, reference, (x, y, w, h)) > threshold:
            return True
    return False


def _frame_number(path: Path) -> int:
    return int(re.search(r"(\d+)$", path.stem).group(1))


def deduplicate_frames(
    frame_dir: str,
    threshold: float = CHANGE_THRESHOLD,
    dry_run: bool = False,
    quiet: bool = False,
    hud=None,
    max_gap_sec: float | None = None,
    fps: float = 2.0,
) -> tuple[int, int]:
    """
    Remove near-duplicate frames from a directory of extracted JPEGs.

    Args:
        frame_dir: Path to directory containing frame_XXXXXX.jpg files.
        threshold: MAD threshold for change detection.
        dry_run: If True, report what would be deleted without deleting.
        hud: Optional src.vision.hud.HudDetector. When given, frames with the
            bottom bar hidden only count as changed on a stage change.
        max_gap_sec: If set, always keep a frame once this much time has
            passed since the last kept one (time from the frame number).
        fps: FPS the frames were extracted at.

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
    ref_number = 0
    ref_bar: bool | None = None   # was the bottom bar visible in the reference frame
    max_gap = max_gap_sec * fps if max_gap_sec is not None else None

    for path in tqdm(paths, desc="Deduplicating", unit="frame", dynamic_ncols=True, disable=quiet,
                     smoothing=0, mininterval=1.0):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        number = _frame_number(path)
        state = hud.detect(frame) if hud is not None else None
        bar = state.shop_visible if state is not None else None
        dx = state.bottom_dx if state is not None else 0
        regions = ["stage"] if bar is False and ref_bar is False else None

        if (reference is None
                or bar != ref_bar
                or (max_gap is not None and number - ref_number >= max_gap)
                or _has_changed(frame, reference, threshold, regions, dx)):
            reference, ref_number, ref_bar = frame, number, bar
            kept += 1
        else:
            if not dry_run:
                path.unlink()
            deleted += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"  Kept {kept} | {action} {deleted} | "
          f"{deleted / max(1, kept + deleted) * 100:.0f}% removed")
    return kept, deleted


def frame_folders(root: str) -> list[Path]:
    """Folders under root (inclusive) that directly hold frames, skipping lobby/."""
    root_path = Path(root)
    folders = {p.parent for p in root_path.rglob("frame_*.jpg")}
    return sorted(f for f in folders if "lobby" not in f.relative_to(root_path).parts)
