"""
Move aside unlabelled unit crops whose position has no health bar by today's
detector: leftovers from before the bar finder.

Crops of the first VODs were collected with an older occupancy test (a fixed
band at the top of each window), which also fired on green spell effects, the
cursor and Little Legend nameplates. recut_unit_crops.py later re-cut them to
the bar-anchored window where a bar was found, but kept the old fixed window
where none was. So the labeling queue held many "not a unit" crops that the
pipeline would never cut today (Hamza, 2026-11-19).

Usage:
    python scripts/prune_stale_unit_crops.py --dry-run     # count only
    python scripts/prune_stale_unit_crops.py

For every unlabelled crop whose frame is still on disk, the crop's hex or
bench slot is checked with board_bar / bench_bar (what unit_reader.occupied
uses). No bar there: the image moves to the removed folder (paths.CROP_REMOVED,
not deleted) and its index.json entry is dropped. Labelled crops are never
touched. Crops whose frame is gone (VODs slimmed after processing) are kept:
they were collected with the bar finder.

Close label_units.py / label_champions.py first: they hold index.json in memory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import shutil
from collections import defaultdict

import cv2
from tqdm import tqdm

from src.utils import paths
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import bench_bar, board_bar

CROP_DIR = paths.CROP_DIR
IMAGES = paths.CROP_IMAGES


def has_bar(frame, name: str, board: BoardHexes, bench: BenchSlots) -> bool:
    pos = re.search(r"__([hs])(\d+)\.jpg$", name)
    if pos is None:     # propagated (__p), rider (h..r) and pair (__top/__bot) crops: never pruned
        return True
    kind, num = pos.group(1), pos.group(2)
    if kind == "h":
        return board_bar(frame, board, int(num[0]), int(num[1])) is not None
    return bench_bar(frame, bench, int(num)) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report only, move nothing")
    args = parser.parse_args()

    index_path, labels_path = CROP_DIR / "index.json", CROP_DIR / "labels.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    board, bench = BoardHexes(), BenchSlots()

    by_frame = defaultdict(list)            # read each frame once
    for name, info in index.items():
        if name not in labels and (IMAGES / name).exists():
            by_frame[info["frame_path"]].append(name)

    stale, no_frame, checked = [], 0, 0
    per_vod = defaultdict(lambda: [0, 0])   # vod -> [checked, stale]
    for frame_path, names in tqdm(sorted(by_frame.items()), unit="frame", dynamic_ncols=True, mininterval=1.0):
        frame = cv2.imread(frame_path)
        if frame is None:
            no_frame += len(names)
            continue
        for name in names:
            checked += 1
            vod = name.split("__")[0]
            per_vod[vod][0] += 1
            if not has_bar(frame, name, board, bench):
                stale.append(name)
                per_vod[vod][1] += 1
    print(f"{checked} unlabelled crops checked ({no_frame} kept: frame no longer on disk); "
          f"{len(stale)} have no health bar at their position")
    for vod, (n, s) in sorted(per_vod.items()):
        print(f"  {vod}: {s} of {n}")
    if args.dry_run or not stale:
        return

    removed = paths.CROP_REMOVED
    removed.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))       # as it is now
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    moved = 0
    for name in stale:
        if name in labels:               # labelled meanwhile: leave it
            continue
        src = IMAGES / name
        if src.exists():
            shutil.move(str(src), str(removed / name))
        index.pop(name, None)
        moved += 1
    index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"Moved {moved} crops to {removed}; {len(index)} crops left in the index")


if __name__ == "__main__":
    main()
