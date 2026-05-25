"""
CLI script: extract frames from all VODs in data/raw/vods/ into data/raw/frames/.

Usage:
    python scripts/extract_frames.py [--fps 1.0]
"""

import argparse
from pathlib import Path
from src.data_collection.frame_extractor import extract_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    vod_dir = Path("data/raw/vods")
    frame_dir = Path("data/raw/frames")

    for vod in vod_dir.glob("*.mp4"):
        out = frame_dir / vod.stem
        print(f"Extracting {vod.name} -> {out}")
        saved = extract_frames(str(vod), str(out), fps=args.fps)
        print(f"  Saved {len(saved)} frames")


if __name__ == "__main__":
    main()
