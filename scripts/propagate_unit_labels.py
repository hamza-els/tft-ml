"""
More examples of a rare unit from the labels it already has: the same unit at
the same hex/slot in the game's other planning frames.

collect_unit_crops.py saves a unit only when it looks different from the last
crop at that position, so a unit that stands still for ten rounds gives one
crop. For a rare unit that is too few to learn from (Big Furry Friend: 3 crops
in 3 games, 2026-11-19). This cuts the same position from every own-board
planning frame of the labelled crop's game, keeps the cuts that still look
like the labelled crop (it may have moved or been sold), and gives them its
label.

Usage:
    python scripts/propagate_unit_labels.py --label "Big Furry Friend"
    python scripts/propagate_unit_labels.py --label "Big Furry Friend" --dry-run

New crops are named like collected ones plus "__p" (…__f012345__h04__p.jpg),
written to the crop images folder, added to index.json with "propagated_from",
and labelled in labels.json. A cut is kept when there is a health bar at the
position (bar-anchored window, as the pipeline cuts) and its fingerprint
correlates with the labelled crop's at >= --min-alike. Check a few in
label_units.py --review before training on them.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re

import cv2
import numpy as np
from tqdm import tqdm

from scripts.collect_unit_crops import _fingerprint
from src.utils import paths
from src.vision.arena import home_board_rows, planning_rows
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import bench_bar, board_bar

CROP_DIR, IMAGES = paths.CROP_DIR, paths.CROP_IMAGES


def cut(frame, wid: str, board: BoardHexes, bench: BenchSlots):
    """The bar-anchored crop at a position, or None when no health bar is there."""
    if wid[0] == "h":
        bar = board_bar(frame, board, int(wid[1]), int(wid[2]))
        win = board.window(int(wid[1]), int(wid[2]), bar) if bar is not None else None
    else:
        bar = bench_bar(frame, bench, int(wid[1:]))
        win = bench.window(int(wid[1:]), bar) if bar is not None else None
    if win is None:
        return None, None
    x, y, w, h = win
    return frame[max(0, y):y + h, max(0, x):x + w], [x, y, w, h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--min-alike", type=float, default=0.75,
                        help="Fingerprint correlation with the labelled crop to keep a cut")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index = json.loads((CROP_DIR / "index.json").read_text(encoding="utf-8"))
    labels = json.loads((CROP_DIR / "labels.json").read_text(encoding="utf-8"))
    sources = [n for n, l in labels.items() if l == args.label and "__p" not in n]
    board, bench = BoardHexes(), BenchSlots()
    new_index, new_labels = {}, {}
    for src in sources:
        m = re.match(r"(.+?)__(game_\d+)__f(\d+)__([hs]\d+)\.jpg$", src)
        img = cv2.imread(str(IMAGES / src))
        if not m or img is None:
            continue
        vod, game, wid = m.group(1), m.group(2), m.group(4)
        ref = _fingerprint(img)
        states = paths.STATES / vod / f"{game}.jsonl"
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        keep = [r for r, h, p in zip(rows, home_board_rows(rows), planning_rows(rows)) if h and p]
        frame_dir = paths.FRAMES / vod / game
        added = 0
        for r in tqdm(keep, desc=f"{vod} {game} {wid}", unit="frame", dynamic_ncols=True, mininterval=1.0):
            name = f"{vod}__{game}__f{r['frame']:06d}__{wid}__p.jpg"
            if name in index or f"{vod}__{game}__f{r['frame']:06d}__{wid}.jpg" in index:
                continue
            frame = cv2.imread(str(frame_dir / f"frame_{r['frame']:06d}.jpg"))
            if frame is None:
                continue
            crop, window = cut(frame, wid, board, bench)
            if crop is None or crop.size == 0 or float(_fingerprint(crop) @ ref) < args.min_alike:
                continue
            new_index[name] = {"frame_path": str(frame_dir / f"frame_{r['frame']:06d}.jpg"), "window": window,
                               "stage": r.get("stage"), "round": r.get("round_num"),
                               "where": "bench" if wid[0] == "s" else "board", "propagated_from": src}
            new_labels[name] = args.label
            if not args.dry_run:
                cv2.imwrite(str(IMAGES / name), crop)
            added += 1
        print(f"  {src}: {added} more crops")
    print(f"{len(new_labels)} new '{args.label}' crops from {len(sources)} labelled ones"
          + (" (dry run, nothing written)" if args.dry_run else ""))
    if args.dry_run or not new_labels:
        return
    # merge into the files as they are now (the labeler may be saving meanwhile)
    index = json.loads((CROP_DIR / "index.json").read_text(encoding="utf-8"))
    labels = json.loads((CROP_DIR / "labels.json").read_text(encoding="utf-8"))
    index.update(new_index)
    labels.update(new_labels)
    (CROP_DIR / "index.json").write_text(json.dumps(index, indent=0), encoding="utf-8")
    (CROP_DIR / "labels.json").write_text(json.dumps(labels, indent=0, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
