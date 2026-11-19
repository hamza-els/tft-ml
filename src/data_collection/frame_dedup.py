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
  - Combat-aware (2026-11-19, Hamza): in combat the bench is not watched;
    gold and the shop still are, and they change when a unit is bought.
  - Board-aware (2026-11-19, Hamza): in planning, a change in which hexes
    hold a unit keeps a frame, at most one every 2 s.
  - Bench, shop and item column are compared per slot, not as one strip.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.vision.arena import is_planning


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
# Watched in combat (combat_aware): not the bench (see deduplicate_frames).
COMBAT_REGIONS = ["gold", "shop", "items", "stage"]

# Regions compared cell by cell, so one slot changing isn't averaged away over
# the whole strip (one bench slot is ~11% of the bench strip): (cells, axis).
# The strips divide evenly into their slots: bench 9 x ~115 px (pitch 116.5),
# shop 5 cards x 202 px, item column 9 slots x ~54 px.
CELLS = {"bench": (9, "x"), "shop": (5, "x"), "items": (9, "y")}


def _cells(region: tuple[int, int, int, int], n: int, axis: str) -> list[tuple[int, int, int, int]]:
    x, y, w, h = region
    if axis == "x":
        return [(x + round(i * w / n), y, round((i + 1) * w / n) - round(i * w / n), h) for i in range(n)]
    return [(x, y + round(i * h / n), w, round((i + 1) * h / n) - round(i * h / n)) for i in range(n)]


def _has_changed(
    frame: np.ndarray,
    reference: np.ndarray,
    threshold: float = CHANGE_THRESHOLD,
    regions: list[str] | None = None,
    bottom_dx: int = 0,
) -> bool:
    """
    Return True if any watched region differs enough from the reference frame.
    Bench, shop and item column are compared per slot: one slot changing counts.
    """
    names = regions if regions is not None else list(WATCH_REGIONS)
    for name in names:
        x, y, w, h = WATCH_REGIONS[name]
        if name in BOTTOM_BAR:
            x += bottom_dx
        n, axis = CELLS.get(name, (1, "x"))
        for cell in _cells((x, y, w, h), n, axis):
            if _region_diff(frame, reference, cell) > threshold:
                return True
    return False


def board_signature(frame: np.ndarray, board=None) -> tuple[bool, ...]:
    """
    Which of the player's 28 hexes hold a unit (a health bar with a star badge
    over it, src/vision/unit_bar.board_bar). Idle animations, effects and the
    Little Legend walking around don't change it; placing, removing or moving
    a unit does. ~18 ms per frame.
    """
    from src.vision.board import BoardHexes
    from src.vision.unit_bar import board_bar
    board = board or _BOARD.setdefault("b", BoardHexes())
    return tuple(board_bar(frame, board, r, c) is not None for r, c in board.hexes())


_BOARD: dict = {}


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
    combat_aware: bool = False,
    watch_board: bool = False,
    board_min_gap_sec: float = 2.0,
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
        combat_aware: In combat (src/vision/arena.is_planning is False), don't
            watch the bench: only gold, shop, item column and stage count.
        watch_board: In planning, also keep a frame when the hexes holding a
            unit change (board_signature): positioning and placing units.
        board_min_gap_sec: Board-triggered keeps are at least this far apart
            (Hamza, 2026-11-19: finicky shuffling isn't important, the settled
            board is; more board frames also help when a tooltip blocks one).

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
    ref_board: tuple | None = None          # hex occupancy at the last kept planning frame
    last_board_keep: int | None = None      # frame number of the last board-triggered keep
    board_gap = board_min_gap_sec * fps
    board_kept = 0

    for path in tqdm(paths, desc="Deduplicating", unit="frame", dynamic_ncols=True, disable=quiet,
                     smoothing=0, mininterval=1.0):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        number = _frame_number(path)
        state = hud.detect(frame) if hud is not None else None
        bar = state.shop_visible if state is not None else None
        dx = state.bottom_dx if state is not None else 0
        planning = is_planning(frame) if (combat_aware or watch_board) else None
        if bar is False and ref_bar is False:
            regions = ["stage"]
        elif combat_aware and not planning:
            # In combat the bench strip mostly shows fighting units and spell
            # effects spilling over from the board (44% of kept combat frames
            # were kept for that alone). A real bench change in combat - a unit
            # bought - also changes gold and the shop, which are still watched.
            regions = COMBAT_REGIONS
        else:
            regions = None

        keep = (reference is None
                or bar != ref_bar
                or (max_gap is not None and number - ref_number >= max_gap)
                or _has_changed(frame, reference, threshold, regions, dx))
        board_now = None
        if watch_board and planning and bar is not False:
            board_now = board_signature(frame)
            if ref_board is None:
                ref_board = board_now          # first planning look: the baseline
            elif (not keep and board_now != ref_board
                  and (last_board_keep is None or number - last_board_keep >= board_gap)):
                # A unit placed, removed or moved. At most one such frame per
                # board_gap: while units are being shuffled the change stays
                # pending against the last kept frame, so the settled board is
                # what gets kept, not every step of the shuffle.
                keep = True
                last_board_keep = number
                board_kept += 1
        if keep:
            reference, ref_number, ref_bar = frame, number, bar
            ref_board = board_now if board_now is not None else (ref_board if planning else None)
            kept += 1
        else:
            if not dry_run:
                path.unlink()
            deleted += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"  Kept {kept} | {action} {deleted} | "
          f"{deleted / max(1, kept + deleted) * 100:.0f}% removed"
          + (f" | {board_kept} kept for a board change" if watch_board else ""))
    return kept, deleted


def frame_folders(root: str) -> list[Path]:
    """Folders under root (inclusive) that directly hold frames, skipping lobby/."""
    root_path = Path(root)
    folders = {p.parent for p in root_path.rglob("frame_*.jpg")}
    return sorted(f for f in folders if "lobby" not in f.relative_to(root_path).parts)
