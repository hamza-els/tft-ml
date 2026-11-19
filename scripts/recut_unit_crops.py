"""
Re-cut every unit crop from its source frame with the current window settings
(src/vision/board.py window(): anchored to the unit's health bar when one is
found, else the fixed window from configs/vision_config.yaml).

Usage:
    python scripts/recut_unit_crops.py

Crop file names stay the same, so labels in labels.json stay attached. Crops
whose source frame is no longer on disk are left as they are (and listed).
Retrain the classifier afterwards (scripts/train_unit_classifier.py): it was
trained on the old window shape.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
from collections import defaultdict

import cv2
from tqdm import tqdm

from src.utils import paths
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import bench_bar, board_bar

CROP_DIR = paths.CROP_DIR          # index.json, labels.json (repo)
IMAGES = paths.CROP_IMAGES         # crop images (local disk, src/utils/paths.py)


def window_for(frame, name: str, board: BoardHexes, bench: BenchSlots) -> tuple[int, int, int, int]:
    """Anchored to the unit's health bar when one is found there, else the fixed window."""
    pos = re.search(r"__([hs])(\d+)\.jpg$", name)
    kind, num = pos.group(1), pos.group(2)
    if kind == "h":
        r, c = int(num[0]), int(num[1])
        return board.window(r, c, board_bar(frame, board, r, c))
    return bench.window(int(num), bench_bar(frame, bench, int(num)))


def main():
    index_path = CROP_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    board, bench = BoardHexes(), BenchSlots()

    by_frame = defaultdict(list)             # read each frame once
    for name, info in index.items():
        if (IMAGES / name).exists():
            by_frame[info["frame_path"]].append(name)

    new_windows, missing = {}, []
    for frame_path, names in tqdm(sorted(by_frame.items()), unit="frame", dynamic_ncols=True, mininterval=1.0):
        frame = cv2.imread(frame_path)
        if frame is None:
            missing.extend(names)
            continue
        for name in names:
            x, y, w, h = window_for(frame, name, board, bench)
            cv2.imwrite(str(IMAGES / name), frame[max(0, y):y + h, max(0, x):x + w])
            new_windows[name] = [x, y, w, h]

    # Re-read before writing: another script may have added crops meanwhile.
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name, window in new_windows.items():
        if name in index:
            index[name]["window"] = window
    index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"Re-cut {len(new_windows)} crops.")
    if missing:
        print(f"{len(missing)} crops kept as they were (source frame gone), e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
