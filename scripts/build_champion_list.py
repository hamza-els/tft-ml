"""
Build assets/champions.txt from the shop itself: OCR the shop card names on a
sample of frames and keep every name seen often enough to not be a misread.

Usage:
    python scripts/build_champion_list.py --frames ~/tft_frames_temp/VOD_ID
    python scripts/build_champion_list.py --frames ~/tft_frames_temp/VOD_ID --samples 1500 --min-count 4

Every unit that can appear in the shop shows up within a few games, so this
gives the current set's roster without a hand-typed list. Review the printed
low-count names: real but rare units (5-costs) sit near the cutoff.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import Counter

import cv2
from tqdm import tqdm

from src.data_collection.frame_dedup import frame_folders
from src.data_collection.label_sampler import pick_evenly
from src.utils.config import vision_cfg
from src.vision import ocr_pipeline
from src.vision.hud import HudDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--samples", type=int, default=1200, help="Frames to OCR in total")
    parser.add_argument("--min-count", type=int, default=4,
                        help="Keep names read at least this many times")
    parser.add_argument("--out", default=vision_cfg["ocr"]["champion_list"])
    args = parser.parse_args()

    ocr_pipeline.CHAMPIONS = []   # read raw names, don't snap to an older list
    hud = HudDetector()
    folders = frame_folders(str(Path(args.frames).expanduser()))
    per_folder = max(1, args.samples // max(1, len(folders)))

    counts: Counter[str] = Counter()
    for folder in folders:
        paths = pick_evenly(sorted(folder.glob("frame_*.jpg")), per_folder)
        for path in tqdm(paths, desc=folder.name, unit="frame", dynamic_ncols=True,
                         smoothing=0, mininterval=1.0):
            frame = cv2.imread(str(path))
            if frame is None or not hud.detect(frame).shop_visible:
                continue
            counts.update(name for name in ocr_pipeline.read_shop(frame) if name)

    kept = sorted(name for name, c in counts.items() if c >= args.min_count)
    dropped = sorted(((c, name) for name, c in counts.items() if c < args.min_count), reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"\n{len(kept)} names -> {out}")
    print("Lowest-count kept:", sorted((counts[n], n) for n in kept)[:15])
    print("Dropped (likely misreads):", dropped[:40])


if __name__ == "__main__":
    main()
