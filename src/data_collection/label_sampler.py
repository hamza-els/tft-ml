"""
Pick a spread-out subset of extracted frames to hand-label for YOLO training.

Consecutive frames are near-identical, so labeling them wastes effort. Frames
are instead sampled evenly across each source folder (one VOD folder, or one
game_XX subfolder after split_frames.py), skipping frames where the bottom
bar isn't visible (lobby, loading, carousel, augment choice).

Copies are named by joining the path relative to the frame root with "__":
    <vod_id>__frame_006383.jpg
    <vod_id>__game_00__frame_006383.jpg
so names stay unique across VODs and the VOD can be recovered for the
train/val split (see src/vision/yolo_dataset.py).
"""

import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.vision.hud import HudDetector

SEP = "__"


def find_sources(root: Path) -> list[Path]:
    """Folders under root that directly contain frames, excluding lobby folders."""
    sources = {p.parent for p in root.rglob("frame_*.jpg")}
    return sorted(s for s in sources if "lobby" not in s.relative_to(root).parts)


def pick_evenly(items: list, n: int) -> list:
    """Return n items evenly spaced across the list (all of them if n >= len)."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def labeled_name(frame_path: Path, root: Path) -> str:
    return SEP.join(frame_path.relative_to(root).parts)


def sample_frames(
    root: str,
    out_dir: str,
    total: int,
    hud: HudDetector | None = None,
) -> list[Path]:
    """
    Copy about `total` in-game frames from under `root` into `out_dir`.

    Args:
        root: Frame root, e.g. ~/tft_frames_temp.
        out_dir: Where labeling images go, e.g. data/processed/labeled/images.
        total: Target number of frames across all sources.
        hud: HUD detector (default: built from vision_config.yaml).

    Returns:
        Paths of newly copied frames (already-present copies are skipped).
    """
    hud = hud or HudDetector()
    root_path = Path(root)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sources = find_sources(root_path)
    if not sources:
        print(f"No frames found under {root_path}")
        return []

    per_source = math.ceil(total / len(sources))
    copied: list[Path] = []

    for source in sources:
        paths = sorted(source.glob("frame_*.jpg"))
        # Oversample, then drop lobby frames — checking every frame is too slow.
        candidates = pick_evenly(paths, per_source * 2)
        in_game = []
        for p in candidates:
            frame = cv2.imread(str(p))
            if frame is not None and hud.detect(frame).shop_visible:
                in_game.append(p)
        chosen = pick_evenly(in_game, per_source)

        new = 0
        for p in chosen:
            dest = out_path / labeled_name(p, root_path)
            if dest.exists():
                continue
            shutil.copy2(p, dest)
            copied.append(dest)
            new += 1
        print(f"  {source.relative_to(root_path)}: "
              f"{len(in_game)}/{len(candidates)} in-game, copied {new}")

    return copied
