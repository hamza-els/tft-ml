"""
Copy a spread-out sample of extracted frames into the labeling folder.

Usage:
    python scripts/sample_for_labeling.py
    python scripts/sample_for_labeling.py --total 300
    python scripts/sample_for_labeling.py --root ~/tft_frames_temp --out data/processed/labeled/images

Then label board + bench champions (name only, no star level) in Roboflow,
CVAT, or LabelImg and export in YOLO format into data/processed/labeled/
(labels/<stem>.txt + classes.txt). Shop cards are read by OCR — don't label them.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.data_collection.label_sampler import sample_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.home() / "tft_frames_temp"),
                        help="Frame root (default: ~/tft_frames_temp)")
    parser.add_argument("--out", default="data/processed/labeled/images")
    parser.add_argument("--total", type=int, default=300,
                        help="Target number of frames across all VODs (default: 300)")
    args = parser.parse_args()

    copied = sample_frames(str(Path(args.root).expanduser()), args.out, args.total)
    print(f"\nCopied {len(copied)} new frame(s) → {args.out}")


if __name__ == "__main__":
    main()
