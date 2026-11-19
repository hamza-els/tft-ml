"""
Retrain every recognition model and print one report card.

Usage:
    python scripts/train_all.py                 # retrain units and items, score stars, report
    python scripts/train_all.py --report-only   # no training: star score and labeler scoreboard

What it does:
  1. Unit classifier (scripts/train_unit_classifier.py): held-out game check, then the final model.
  2. Item classifier (scripts/train_item_classifier.py): matcher vs model vs hybrid on held-out
     games, then the final model.
  3. Star reader (template-based, src/vision/unit_bar.star_level): scored against the star
     labels from scripts/label_champions.py.
  4. Scoreboard from label_champions.py: how often each model's pre-filled answer was kept.

The full logs go to ~/tft_data/logs/train_all.log; the report card is printed at the end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter

import cv2

from scripts.label_champions import REVIEWS, STAR_LABELS, load, scoreboard
from src.utils import paths
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import bench_bar, board_bar, star_level

ROOT = Path(__file__).resolve().parents[1]
LOG = paths.LOGS / "train_all.log"


def run(script: str) -> str:
    """Run a training script, log everything, return its output."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t = time.time()
    out = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env)
    text = out.stdout + out.stderr
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {script} ({(time.time() - t) / 60:.0f} min)\n{text}")
    if out.returncode != 0:
        return f"FAILED (exit {out.returncode}); see {LOG}"
    return text


def unit_summary(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines()]
    out = [l for l in lines if l.startswith("Right first guess") or "labelled crops" in l]
    per = [l for l in lines if re.match(r"^\S.*\s\d+/\d+\s+\(\d+ in training\)$", l)]
    weak = sorted(per, key=lambda l: int(re.search(r"(\d+)/(\d+)", l).group(1)) / max(1, int(re.search(r"(\d+)/(\d+)", l).group(2))))[:5]
    return out + (["weakest: " + "; ".join(weak)] if weak else [])


def item_summary(text: str) -> list[str]:
    lines = text.splitlines()
    if "On held-out games:" not in text:
        return [text.strip()[-300:]]
    k = next(i for i, l in enumerate(lines) if l.startswith("On held-out games"))
    return [l.strip() for l in lines[k + 1:k + 4] if l.strip()]


def star_score() -> list[str]:
    labels = load(STAR_LABELS)
    if not labels:
        return ["no star labels yet (scripts/label_champions.py)"]
    index = load(paths.CROP_DIR / "index.json")
    board, bench = BoardHexes(), BenchSlots()
    confusion, unread, right, n = Counter(), 0, 0, 0
    for name, star in labels.items():
        info = index.get(name)
        frame = cv2.imread(info["frame_path"]) if info else None
        if frame is None:
            continue
        pos = re.search(r"__([hs]\d+)\.jpg$", name).group(1)
        bar = (board_bar(frame, board, int(pos[1]), int(pos[2])) if pos[0] == "h"
               else bench_bar(frame, bench, int(pos[1:])))
        read = star_level(frame, bar) if bar else None
        n += 1
        if read is None:
            unread += 1
        elif read == star:
            right += 1
        else:
            confusion[(star, read)] += 1
    lines = [f"{right}/{n} right ({right / max(n, 1):.1%}), {unread} unread"]
    if confusion:
        lines.append("mix-ups (label -> read): " + ", ".join(f"{a}->{b} x{c}" for (a, b), c in confusion.most_common()))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true", help="Skip training")
    args = parser.parse_args()

    card: list[str] = []
    if not args.report_only:
        print("Training the unit classifier ...", flush=True)
        card += ["Units (held-out game):"] + [f"  {l}" for l in unit_summary(run("scripts/train_unit_classifier.py"))]
        print("Training the item classifier ...", flush=True)
        card += ["Items (held-out games):"] + [f"  {l}" for l in item_summary(run("scripts/train_item_classifier.py"))]
    print("Scoring the star reader ...", flush=True)
    card += ["Stars (template reader vs star labels):"] + [f"  {l}" for l in star_score()]
    s = scoreboard(load(REVIEWS))
    card += ["Champion labeler scoreboard (models' answers kept unchanged):"]
    card += [f"  {k:11s} {a}/{n} ({a / n:.0%})" if n else f"  {k:11s} -" for k, (a, n) in s.items()]
    print("\n===== Report card =====")
    print("\n".join(card))
    print(f"(full logs: {LOG})")


if __name__ == "__main__":
    main()
