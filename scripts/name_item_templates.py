"""
Name the item templates under units' bars (assets/templates/unit_items/)
from Hamza's item labels and from the official item art.

Usage:
    python scripts/name_item_templates.py --dry-run
    python scripts/name_item_templates.py

1. From labels (data/processed/item_crops/labels.json): a template whose
   labelled crops (crops the matcher assigned to it) agree on one item -
   at least MIN_LABELS crops, LABEL_AGREE of them the same - takes that name.
   This is how "Radiant Spirit Visage" became Spirit Visage: the automatic
   naming can't tell Radiant from normal at 25 px, Hamza can.
2. From the official art (assets/templates/items/, Community Dragon - the
   same icons sites like metatft.com show): each still unnamed template is
   matched against every item icon of the set with the item matcher
   (src/vision/unit_bar.item_similarity). The score needed to trust the
   name is measured on the templates the labels named: the lowest score at
   which the official-art guess was right on OFFICIAL_PRECISION of them.

Files are renamed in place (Name.png, Name__2.png, ...); the old folder is
copied to ~/tft_data/backup/ first.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.items import _ROOT
from src.vision.unit_bar import item_hue, item_vec

FOLDER = _ROOT / vision_cfg["unit_bar"]["item_templates"]
OFFICIAL = _ROOT / vision_cfg["template_matching"]["item_dir"]
SIZE = vision_cfg["unit_bar"]["item_size"]
MIN_LABELS = 3
LABEL_AGREE = 0.8
OFFICIAL_PRECISION = 0.95
APPLY_OFFICIAL = False    # official-art names are proposed for review (make_item_review.py --official), not applied


def features(img):
    return item_vec(img), item_hue(img)


def sim(a, b) -> float:
    return float(a[0] @ b[0]) * float(a[1] @ b[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(FOLDER.glob("*.png"))
    ids = [f.stem.split("__")[0] for f in files]
    index = json.loads((paths.ITEM_DIR / "index.json").read_text(encoding="utf-8"))
    labels = json.loads((paths.ITEM_DIR / "labels.json").read_text(encoding="utf-8"))

    # 1. Labels: which item did Hamza say each template's crops show?
    by_guess: dict[str, Counter] = defaultdict(Counter)
    for crop, label in labels.items():
        guess = index.get(crop, {}).get("guess")
        if guess and not label.startswith("_"):
            by_guess[guess][label] += 1
    final = {i: i for i in set(ids)}
    from_labels = {}
    for tid, counts in by_guess.items():
        label, n = counts.most_common(1)[0]
        total = sum(counts.values())
        if tid in final and total >= MIN_LABELS and n / total >= LABEL_AGREE:
            from_labels[tid] = label
            final[tid] = label

    # 2. Official art for templates still unnamed.
    official = {}
    for p in sorted(OFFICIAL.glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            official[p.stem] = features(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    tmpl = {}
    for f, i in zip(files, ids):
        tmpl.setdefault(i, features(cv2.imread(str(f))))

    def best_official(i):
        scored = sorted(((sim(tmpl[i], v), n) for n, v in official.items()), reverse=True)
        return scored[0][1], scored[0][0], scored[1][0] if len(scored) > 1 else 0.0

    # Calibrate on templates whose name the labels confirmed.
    checks = sorted(((best_official(t)[1], best_official(t)[0] == name) for t, name in from_labels.items()),
                    reverse=True)
    threshold = None
    for k in range(len(checks), 0, -1):
        top = checks[:k]
        if sum(ok for _, ok in top) / len(top) >= OFFICIAL_PRECISION:
            threshold = top[-1][0]
            break
    right = sum(ok for _, ok in checks)
    print(f"Official-art naming checked on {len(checks)} label-named templates: right on {right}; "
          f"{OFFICIAL_PRECISION:.0%} right from score {threshold}")

    from_official = {}
    for i in sorted(set(ids)):
        if APPLY_OFFICIAL and i.startswith("uitem_") and i not in from_labels and threshold is not None:
            name, score, second = best_official(i)
            if score >= threshold:
                from_official[i] = (name, score)
                final[i] = name

    print(f"\nFrom labels ({len(from_labels)}):")
    for t, n in sorted(from_labels.items()):
        if t != n:
            print(f"  {t:30s} -> {n}")
    print(f"\nFrom official art ({len(from_official)}):")
    for t, (n, s) in sorted(from_official.items()):
        print(f"  {t:30s} -> {n}  ({s:.2f})")
    still = sorted(i for i in set(ids) if final[i].startswith("uitem_"))
    print(f"\n{len(set(ids))} IDs before, {len(set(final.values()))} after; {len(still)} still unnamed")
    if args.dry_run:
        return

    backup = paths.LOCAL / "backup" / f"unit_items_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(FOLDER, backup)
    data = [(f.read_bytes(), final[i]) for f, i in zip(files, ids)]
    for f in files:
        f.unlink()
    used: Counter = Counter()
    for blob, name in data:
        used[name] += 1
        stem = name if used[name] == 1 else f"{name}__{used[name]}"
        (FOLDER / f"{stem}.png").write_bytes(blob)
    print(f"Renamed; previous templates in {backup}")


if __name__ == "__main__":
    main()
