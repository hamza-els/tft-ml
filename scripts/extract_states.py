"""
Run the vision pipeline over game-split frames and write one StateVector per
frame as JSON lines.

Usage:
    python scripts/extract_states.py --frames ~/tft_frames_temp/VOD_ID
    python scripts/extract_states.py --frames ~/tft_frames_temp/VOD_ID --games game_05
    python scripts/extract_states.py --frames ~/tft_frames_temp/VOD_ID --limit 200

Output (local disk, see src/utils/paths.py):
    ~/tft_data/state_vectors/<VOD_ID>/game_XX.jsonl
    Each line: {"frame": 30059, "time_sec": 15029.5, ...StateVector fields}

Board and bench stay empty until assets/yolo_tft.pt exists (scripts/train_yolo.py).
Games whose output file already exists are skipped unless --overwrite is given.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import cv2
from tqdm import tqdm

from src.data_collection.frame_dedup import frame_folders
from src.data_collection.frame_game_splitter import frame_number
from src.utils import paths
from src.vision.state_extractor import StateExtractor

OUT_DIR = paths.STATES     # local disk; only finished states are copied to the repo


def replace_with_retry(src: Path, dst: Path, tries: int = 30, wait: float = 2.0):
    """
    Rename src over dst. Google Drive's sync briefly locks files it is
    uploading (WinError 32), which once failed a whole VOD; wait it out.
    """
    for i in range(tries):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--games", nargs="*", help="Only these game folders (e.g. game_00 game_03)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="FPS used during frame extraction (default: 2.0)")
    parser.add_argument("--limit", type=int, default=None, help="Max frames per game (for testing)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    vod_dir = Path(args.frames).expanduser()
    extractor = StateExtractor.from_config()
    if extractor.detector is None:
        print("No YOLO weights yet - board/bench will be empty.")

    for folder in frame_folders(str(vod_dir)):
        if args.games and folder.name not in args.games:
            continue
        out = OUT_DIR / vod_dir.name / f"{folder.name}.jsonl"
        if out.exists() and not args.overwrite:
            print(f"{out} exists - skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        paths = sorted(folder.glob("frame_*.jpg"))[:args.limit]
        tmp = out.with_suffix(".jsonl.part")
        extractor.start_sequence()
        with open(tmp, "w", encoding="utf-8") as f:
            for path in tqdm(paths, desc=folder.name, unit="frame", dynamic_ncols=True,
                             smoothing=0, mininterval=1.0):
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                n = frame_number(path)
                row = {"frame": n, "time_sec": n / args.fps, **extractor.extract(frame).to_dict()}
                f.write(json.dumps(row) + "\n")
        replace_with_retry(tmp, out)
        c = extractor.ocr_cache
        print(f"  {len(paths)} frames -> {out}  (OCR reads {c.reads}, skipped as unchanged {c.skips})")


if __name__ == "__main__":
    main()
