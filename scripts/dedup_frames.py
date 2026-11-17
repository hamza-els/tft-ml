"""
Remove near-duplicate frames from extracted (and ideally game-split) frames.

Usage:
    # Every game folder of one VOD (lobby/ is skipped)
    python scripts/dedup_frames.py --frames ~/tft_frames_temp/VOD_ID

    # Report what would be removed, delete nothing
    python scripts/dedup_frames.py --frames ~/tft_frames_temp/VOD_ID --dry-run

Run split_frames.py first: it needs the full frame density near game
boundaries, and it moves lobby frames out of the way.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.data_collection.frame_dedup import CHANGE_THRESHOLD, deduplicate_frames, frame_folders
from src.vision.hud import HudDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder (or a single game folder)")
    parser.add_argument("--threshold", type=float, default=CHANGE_THRESHOLD)
    parser.add_argument("--max-gap", type=float, default=5.0,
                        help="Always keep a frame after this many seconds (default: 5)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="FPS used during frame extraction (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hud = HudDetector()
    total_kept = total_deleted = 0
    for folder in frame_folders(str(Path(args.frames).expanduser())):
        print(f"\n{folder}")
        kept, deleted = deduplicate_frames(str(folder), args.threshold, args.dry_run,
                                           hud=hud, max_gap_sec=args.max_gap, fps=args.fps)
        total_kept += kept
        total_deleted += deleted

    print(f"\nTotal: kept {total_kept} | {'would remove' if args.dry_run else 'removed'} {total_deleted}")


if __name__ == "__main__":
    main()
