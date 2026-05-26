"""
CLI script: extract frames from game clip(s) into data/raw/frames/.

Usage:
    # Single file
    python scripts/extract_frames.py --video data/raw/vods/game.mp4

    # All .mp4s in a directory, delete each VOD after extraction to save space
    python scripts/extract_frames.py --video-dir data/raw/vods --delete-after

    # Parallel processing (2-3 workers recommended for Google Drive)
    python scripts/extract_frames.py --video-dir data/raw/vods --workers 2 --delete-after

    # FPS guide:
    #   --fps 0.2  →  1 frame per 5s  — bulk macro training data (recommended)
    #   --fps 1.0  →  1 frame per 1s  — calibration / template collection
"""

import sys
import ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_collection.frame_extractor import extract_frames
from src.data_collection.frame_dedup import deduplicate_frames


ES_CONTINUOUS      = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def _prevent_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def _allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def process_video(
    video: Path,
    frame_dir: Path,
    fps: float,
    delete_after: bool,
    dedup: bool,
) -> tuple[str, int]:
    out = frame_dir / video.stem
    saved = extract_frames(str(video), str(out), fps=fps)

    if dedup:
        kept, _ = deduplicate_frames(str(out))
        saved = kept

    if delete_after:
        video.unlink()
        print(f"  Deleted {video.name}")

    return video.name, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Single video file to process")
    parser.add_argument("--video-dir", default="data/raw/vods",
                        help="Directory of .mp4 files to process")
    parser.add_argument("--fps", type=float, default=0.2,
                        help="Frames per second to extract. "
                             "0.2 = 1 frame per 5s (bulk training). "
                             "1.0 = 1 frame per 1s (calibration/templates). "
                             "Default: 0.2")
    parser.add_argument("--workers", type=int, default=1,
                        help="Videos to process in parallel. "
                             "2-3 recommended for Google Drive. Default: 1")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete each VOD immediately after its frames are "
                             "extracted. Frees Drive space for the next download.")
    parser.add_argument("--dedup", action="store_true",
                        help="Remove near-duplicate frames after extraction. "
                             "Keeps dense frames during rerolls, drops redundant "
                             "combat/idle frames. Recommended for training data.")
    args = parser.parse_args()

    frame_dir = Path("data/raw/frames")

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(Path(args.video_dir).glob("*.mp4"))

    if not videos:
        print("No .mp4 files found.")
        return

    print(f"Found {len(videos)} video(s)")
    print(f"FPS: {args.fps}  |  Workers: {args.workers}  |  "
          f"Delete after: {args.delete_after}  |  Dedup: {args.dedup}")
    print()

    _prevent_sleep()

    try:
        if args.workers == 1 or len(videos) == 1:
            for i, video in enumerate(videos, 1):
                print(f"[{i}/{len(videos)}] {video.name}")
                name, count = process_video(video, frame_dir, args.fps, args.delete_after, args.dedup)
                print(f"  Saved {count} frames\n")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(process_video, v, frame_dir, args.fps, args.delete_after, args.dedup): v
                    for v in videos
                }
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    name, count = future.result()
                    print(f"[{completed}/{len(videos)}] ✓ {name} — {count} frames")
    finally:
        _allow_sleep()
        print("\nAll done.")


if __name__ == "__main__":
    main()
