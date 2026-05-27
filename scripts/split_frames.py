"""
Split already-extracted frames from multi-game streams into per-game subfolders.

Usage:
    # Split a single VOD's frames
    python scripts/split_frames.py --frames data/raw/frames/VOD_ID

    # Split all frame folders that haven't been split yet
    python scripts/split_frames.py --all

    # Delete lobby frames instead of keeping them
    python scripts/split_frames.py --frames data/raw/frames/VOD_ID --delete-lobby

    # If you extracted at a different FPS
    python scripts/split_frames.py --frames data/raw/frames/VOD_ID --fps 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import yaml

from src.data_collection.frame_game_splitter import find_game_segments, split_into_subfolders


FRAME_DIR = Path("data/raw/frames")


def already_split(folder: Path) -> bool:
    """Return True if this folder already has game_XX subfolders."""
    return any(folder.is_dir() and folder.name.startswith("game_")
               for folder in folder.iterdir())


def process_folder(folder: Path, hud_regions: dict, fps: float, delete_lobby: bool):
    print(f"\n{'═' * 60}")
    print(f"  {folder.name}")
    print(f"{'═' * 60}")

    if already_split(folder):
        print("  Already split — skipping. Delete game_XX subfolders to re-run.")
        return

    raw_frames = list(folder.glob("frame_*.jpg"))
    if not raw_frames:
        print("  No frames found — skipping.")
        return

    segments = find_game_segments(
        frame_dir=str(folder),
        hud_regions=hud_regions,
        extraction_fps=fps,
    )

    if not segments:
        print("  No games detected. Check that hud_regions are calibrated.")
        return

    split_into_subfolders(
        frame_dir=str(folder),
        segments=segments,
        move_lobby=not delete_lobby,
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frames", help="Path to a single frame folder to split")
    group.add_argument("--all", action="store_true",
                       help="Split all frame folders in data/raw/frames/")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="FPS used during frame extraction (default: 2.0)")
    parser.add_argument("--delete-lobby", action="store_true",
                        help="Delete lobby/loading frames instead of moving them")
    args = parser.parse_args()

    with open("configs/vision_config.yaml") as f:
        cfg = yaml.safe_load(f)

    regions = cfg["ocr"]["regions"]
    hud_regions = {
        "gold": tuple(regions["gold"]),
        "hp":   tuple(regions["hp"]),
    }

    if args.frames:
        process_folder(Path(args.frames), hud_regions, args.fps, args.delete_lobby)
    else:
        folders = [f for f in sorted(FRAME_DIR.iterdir())
                   if f.is_dir() and not f.name.startswith(".")]
        if not folders:
            print(f"No frame folders found in {FRAME_DIR}")
            return
        print(f"Found {len(folders)} frame folder(s)")
        for folder in folders:
            process_folder(folder, hud_regions, args.fps, args.delete_lobby)

    print(f"\n{'═' * 60}")
    print("  Done.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
