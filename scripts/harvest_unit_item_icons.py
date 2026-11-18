"""
Build templates for the item icons drawn under units' health bars, cut from
footage.

Usage:
    python scripts/harvest_unit_item_icons.py

Goes through every unit crop in data/processed/unit_crops/index.json (their
source frames must still be on disk), finds the unit's health bar, and keeps
each item slot whose dark icon outline is clearly there. Icons that look
alike are grouped; each group's median picture becomes a template in
assets/templates/unit_items/:

    <Item Name>.png    when the group matches a known item icon well
                       (items_seen/ or the official art), else
    uitem_XXX.png      an ID; rename the file to give it a name.

Several templates can stand for one item: <Item Name>__2.png etc. (anything
after "__" is ignored when reading). The same item sometimes splits into
two groups (a stack counter, a Radiant frame); give the extra file the same
name with a __ suffix to make them read as one item.

The model only needs consistent IDs, so unnamed templates are fine.
Running it again rebuilds the folder from scratch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config import vision_cfg
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.items import _ROOT, _TemplateSet
from src.vision.unit_bar import bench_bar, board_bar, crop, item_vec, outline_score

CROP_DIR = Path("data/processed/unit_crops")
OUT = _ROOT / vision_cfg["unit_bar"]["item_templates"]
MIN_OUTLINE = 0.45       # only icons with a visible outline become templates
SAME = 0.85              # correlation above which two icons are the same item
MERGE = 0.85             # groups whose median pictures correlate this well are merged
MIN_MEMBERS = 3          # groups smaller than this are likely glitches
NAME_SCORE = 0.80        # median icon vs known item art needed to take its name
TWIN_SCORE = 0.79       # unnamed group this close to a named one takes its name (checked by eye: all
                         # pairs at >= 0.79 were the same item, the first wrong pair was at 0.78)
KNOWN_SCORE = 0.84       # an outline-less slot is kept when it matches known item art this well


def aligned_icon(frame, region):
    """The icon at the small shift where its outline fits best (ties: the smallest shift)."""
    x, y, w, h = region
    best = max(((outline_score(frame, (x + dx, y + dy, w, h)), -abs(dx) - abs(dy), dx, dy)
                for dx in (-2, -1, 0, 1, 2) for dy in (-1, 0, 1)))
    _, _, dx, dy = best
    return crop(frame, (x + dx, y + dy, w, h))


def main():
    index = json.loads((CROP_DIR / "index.json").read_text(encoding="utf-8"))
    board, bench = BoardHexes(), BenchSlots()
    by_frame: dict[str, list[str]] = {}
    for name, info in index.items():
        by_frame.setdefault(info["frame_path"], []).append(name)

    size = vision_cfg["unit_bar"]["item_size"]
    known = []
    for folder in (vision_cfg["item_column"]["seen_dir"], vision_cfg["template_matching"]["item_dir"]):
        ts = _TemplateSet(size, 2, mask_corner=False)
        ts.load(_ROOT / folder)
        known.append(ts)

    def known_score(icon):
        return max(ts.best(icon)[1] for ts in known)

    icons = []
    for frame_path, names in tqdm(sorted(by_frame.items()), unit="frame", dynamic_ncols=True, mininterval=1.0):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        for name in names:
            kind, num = re.search(r"__([hs])(\d+)\.jpg$", name).groups()
            bar = (board_bar(frame, board, int(num[0]), int(num[1])) if kind == "h"
                   else bench_bar(frame, bench, int(num)))
            if bar is None:
                continue
            for region in bar.item_regions():
                if outline_score(frame, region) >= MIN_OUTLINE:
                    icon = aligned_icon(frame, region)
                else:
                    # Some icons (Void Staff's purple glow) hide their outline:
                    # keep those that clearly look like known item art.
                    x, y, w, h = region
                    icon = max((crop(frame, (x + dx, y + dy, w, h)) for dx in (-2, -1, 0, 1, 2) for dy in (-1, 0, 1)),
                               key=lambda c: known_score(c) if c.shape[:2] == (h, w) else 0)
                    if icon.shape[:2] != (h, w) or known_score(icon) < KNOWN_SCORE:
                        continue
                if icon.shape[:2] == (region[3], region[2]):
                    icons.append(icon.copy())
    print(f"{len(icons)} item icons found under health bars")

    # Greedy grouping against running group centres.
    groups: list[list[int]] = []
    centres: list[np.ndarray] = []
    vecs = [item_vec(i) for i in icons]
    for i, v in enumerate(vecs):
        if centres:
            scores = np.stack(centres) @ v
            k = int(np.argmax(scores))
            if scores[k] >= SAME:
                groups[k].append(i)
                centres[k] = item_vec(np.median([icons[j] for j in groups[k][-50:]], axis=0))
                continue
        groups.append([i])
        centres.append(v)

    # Second pass: greedy grouping depends on order, so merge groups whose
    # median pictures turned out alike (largest first).
    def median_vec(g):
        return item_vec(np.median([icons[j] for j in g[:200]], axis=0))
    groups.sort(key=len, reverse=True)
    merged: list[list[int]] = []
    merged_vecs: list[np.ndarray] = []
    for g in groups:
        v = median_vec(g)
        if merged_vecs:
            scores = np.stack(merged_vecs) @ v
            k = int(np.argmax(scores))
            if scores[k] >= MERGE:
                merged[k].extend(g)
                continue
        merged.append(list(g))
        merged_vecs.append(v)
    # Dark, flat "icons" are scenery that happened to pass the outline check.
    groups = [g for g in merged if len(g) >= MIN_MEMBERS
              and np.median([icons[j] for j in g], axis=0).std() > 25]
    groups.sort(key=len, reverse=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    medians = [np.median([icons[j] for j in g], axis=0).astype(np.uint8) for g in groups]
    names: list[str | None] = []
    for median in medians:
        name, score = max((ts.best(median) for ts in known), key=lambda r: r[1])
        names.append(name if name and score >= NAME_SCORE and not name.startswith("item_") else None)
    # An unnamed group that looks like a named one is another picture of that
    # item (stack counter, Radiant frame, a frame where it was half covered).
    named_ids = [i for i, n in enumerate(names) if n]
    if named_ids:
        named_vecs = np.stack([item_vec(medians[i]) for i in named_ids])
        for i, n in enumerate(names):
            if n is None:
                scores = named_vecs @ item_vec(medians[i])
                k = int(np.argmax(scores))
                if scores[k] >= TWIN_SCORE:
                    names[i] = names[named_ids[k]]

    used: set[str] = set()
    named = 0
    for n, (median, name) in enumerate(zip(medians, names)):
        if name:
            copies = sum(1 for u in used if u == name or u.startswith(name + "__"))
            stem = name if copies == 0 else f"{name}__{copies + 1}"
            used.add(stem)
            named += 1
        else:
            stem = f"uitem_{n:03d}"
        cv2.imwrite(str(OUT / f"{stem}.png"), median)
    print(f"{len(groups)} templates ({named} named) -> {OUT}")


if __name__ == "__main__":
    main()
