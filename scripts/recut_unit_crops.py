"""
Re-cut every unit crop from its source frame with the current window settings
(board / bench window_pad etc. in configs/vision_config.yaml).

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

from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes

CROP_DIR = Path("data/processed/unit_crops")


def window_for(name: str, board: BoardHexes, bench: BenchSlots) -> tuple[int, int, int, int]:
    pos = re.search(r"__([hs])(\d+)\.jpg$", name)
    kind, num = pos.group(1), pos.group(2)
    return board.window(int(num[0]), int(num[1])) if kind == "h" else bench.window(int(num))


def main():
    index_path = CROP_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    board, bench = BoardHexes(), BenchSlots()

    by_frame = defaultdict(list)             # read each frame once
    for name, info in index.items():
        if (CROP_DIR / "images" / name).exists():
            by_frame[info["frame_path"]].append(name)

    new_windows, missing = {}, []
    for frame_path, names in tqdm(sorted(by_frame.items()), unit="frame", dynamic_ncols=True, mininterval=1.0):
        frame = cv2.imread(frame_path)
        if frame is None:
            missing.extend(names)
            continue
        for name in names:
            x, y, w, h = window_for(name, board, bench)
            cv2.imwrite(str(CROP_DIR / "images" / name), frame[max(0, y):y + h, max(0, x):x + w])
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
