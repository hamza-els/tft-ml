"""
Remove unit crops that were cut from combat frames (or from another player's
board), keeping only the player's own board in the planning phase.

Usage:
    python scripts/prune_unit_crops.py --states data/processed/state_vectors/VOD_ID
    python scripts/prune_unit_crops.py --states data/processed/state_vectors/VOD_ID --dry-run

Units don't change once a fight starts, and mid-fight they are off their
hexes, overlapping, or half out of the window — bad examples to learn from.

Removed crops are moved to data/processed/unit_crops/removed/ (not deleted),
and their entries are taken out of index.json and labels.json. Labels already
given to crops that stay are untouched. Labels of removed crops are kept in
removed/labels.json in case they are wanted back.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import shutil

from src.vision.arena import home_board_rows, planning_rows

CROP_DIR = Path("data/processed/unit_crops")


def keep_frames(states_dir: Path) -> dict[str, set[int]]:
    """game name -> frame numbers that are own-board planning frames."""
    out = {}
    for states in sorted(states_dir.glob("game_*.jsonl")):
        if states.name.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
        rows = sorted((r for r in rows if r.get("in_game")), key=lambda r: r["frame"])
        if not any(r.get("timer_fill") is not None for r in rows):
            sys.exit(f"{states.name} has no timer_fill - run scripts/backfill_states.py first")
        home, planning = home_board_rows(rows), planning_rows(rows)
        out[states.stem] = {r["frame"] for r, h, p in zip(rows, home, planning) if h and p}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--dry-run", action="store_true", help="Report only, move nothing")
    args = parser.parse_args()

    keep = keep_frames(Path(args.states))
    index_path, labels_path = CROP_DIR / "index.json", CROP_DIR / "labels.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    remove = []
    for name in index:
        m = re.search(r"__(game_\d+)__f(\d+)__", name)
        if m and m.group(1) in keep and int(m.group(2)) not in keep[m.group(1)]:
            remove.append(name)

    labelled = [n for n in remove if n in labels]
    print(f"{len(index)} crops; {len(remove)} are from combat or away frames "
          f"({len(labelled)} of those already labelled); {len(index) - len(remove)} stay")
    if args.dry_run or not remove:
        return

    removed_dir = CROP_DIR / "removed"
    removed_dir.mkdir(exist_ok=True)
    removed_labels_path = removed_dir / "labels.json"
    removed_labels = json.loads(removed_labels_path.read_text(encoding="utf-8")) \
        if removed_labels_path.exists() else {}
    for name in remove:
        src = CROP_DIR / "images" / name
        if src.exists():
            shutil.move(str(src), str(removed_dir / name))
        del index[name]
        if name in labels:
            removed_labels[name] = labels.pop(name)

    index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
    labels_path.write_text(json.dumps(labels, indent=0, sort_keys=True), encoding="utf-8")
    removed_labels_path.write_text(json.dumps(removed_labels, indent=0, sort_keys=True), encoding="utf-8")
    print(f"Moved {len(remove)} crops to {removed_dir}; {len(labels)} labels remain")


if __name__ == "__main__":
    main()
