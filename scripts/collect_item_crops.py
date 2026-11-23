"""
Cut out the item icons under units' health bars, ready to be labelled with
scripts/label_items.py.

Usage:
    python scripts/collect_item_crops.py                 # every VOD with unit crops
    python scripts/collect_item_crops.py --vod knB6dE8Rbgg

Source: the unit crops (data/processed/unit_crops/index.json) and their
frames. For each unit, its health bar is found (src/vision/unit_bar.py) and
each of the three item slots under it is kept when it might hold an item:
a visible icon outline, or a template match. Slots that only might be items
are kept too, so "not an item" can be labelled. A slot that looks the same
as the last one kept for that unit position is skipped (an item a unit holds
for ten rounds is labelled once, not ten times).

Output:
    ~/tft_data/item_crops/images/<unit crop stem>__i<slot>.png    the icon (25 px + 4 px around)
    ~/tft_data/item_crops/context/<same name>.png                 bar, badge and item row
    data/processed/item_crops/index.json   name -> frame, region, unit crop, matcher guess and score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import paths
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import UnitItemReader, bench_bar, board_bar, crop, outline_score

PAD = 4                  # px of scenery kept around the 25 px icon
MAYBE_OUTLINE = 0.30     # keep a slot with at least this much icon outline...
MAYBE_SCORE = 0.40       # ...or a template match at least this good
SAME = 0.95              # a slot this alike to the last one kept at its position is skipped


def fingerprint(img: np.ndarray) -> np.ndarray:
    v = img.astype(np.float32).reshape(-1)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod", help="Only this VOD")
    args = parser.parse_args()

    unit_index = json.loads((paths.CROP_DIR / "index.json").read_text(encoding="utf-8"))
    index_path = paths.ITEM_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    paths.ITEM_IMAGES.mkdir(parents=True, exist_ok=True)
    paths.ITEM_CONTEXT.mkdir(parents=True, exist_ok=True)
    board, bench, reader = BoardHexes(), BenchSlots(), UnitItemReader()

    by_frame = defaultdict(list)
    for name, info in unit_index.items():
        if args.vod and not name.startswith(args.vod + "__"):
            continue
        by_frame[info["frame_path"]].append(name)

    last: dict[str, np.ndarray] = {}        # "<vod>__<game>__<pos>__i<slot>" -> fingerprint of last kept
    added = skipped = 0
    for frame_path in tqdm(sorted(by_frame, key=lambda p: (Path(p).parent.as_posix(), p)),
                           unit="frame", dynamic_ncols=True, mininterval=1.0):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        for unit_name in by_frame[frame_path]:
            m = re.match(r"(.+?)__(game_\d+)__f(\d+)__([hs]\d+)\.jpg$", unit_name)
            if not m:       # propagated (__p) and rider-pair (__top/__bot) crops: no bar at their own position
                continue
            vod, game, _, pos = m.groups()
            bar = (board_bar(frame, board, int(pos[1]), int(pos[2])) if pos[0] == "h"
                   else bench_bar(frame, bench, int(pos[1:])))
            if bar is None:
                continue
            for slot, region in enumerate(bar.item_regions()):
                outline = outline_score(frame, region)
                guess, score = reader.identify(frame, region)
                if outline < MAYBE_OUTLINE and score < MAYBE_SCORE:
                    continue
                x, y, w, h = region
                icon = crop(frame, (x - PAD, y - PAD, w + 2 * PAD, h + 2 * PAD))
                if icon.shape[:2] != (h + 2 * PAD, w + 2 * PAD):
                    continue
                key = f"{vod}__{game}__{pos}__i{slot}"
                fp = fingerprint(icon)
                if key in last and float(fp @ last[key]) >= SAME:
                    skipped += 1
                    continue
                last[key] = fp
                name = unit_name[:-4] + f"__i{slot}.png"
                if name in index:
                    continue
                cv2.imwrite(str(paths.ITEM_IMAGES / name), icon)
                cv2.imwrite(str(paths.ITEM_CONTEXT / name),
                            crop(frame, (bar.left - 32, bar.top - 16, 130, 56)))
                index[name] = {"frame_path": frame_path, "region": [int(v) for v in region],
                               "unit_crop": unit_name, "slot": slot,
                               "guess": guess, "score": round(score, 3), "outline": round(outline, 3)}
                added += 1

    # Merge into the file as it is now (another run may have added entries).
    fresh = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    fresh.update(index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(fresh, indent=0), encoding="utf-8")
    print(f"{added} new item crops ({skipped} skipped as unchanged); {len(fresh)} in total -> {paths.ITEM_IMAGES}")
    print("Label them with: python scripts/label_items.py")


if __name__ == "__main__":
    main()
