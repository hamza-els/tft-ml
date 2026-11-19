"""
Add the fields that don't need OCR to already-extracted state files, so a new
reader can be applied without re-running the slow extraction.

Usage:
    python scripts/backfill_states.py --frames ~/tft_frames_temp/VOD_ID \
        --states data/processed/state_vectors/VOD_ID

Recomputed per row (the frames must still be on disk):
    arena_sig, planning,       own-board / planning-phase detection
    timer_fill
    augments, augment_tiers    augment bar
    away_augments              other player's augment bar (scouting / away fights)
    bench_items                item column
    hp, other_hp               with --reread-hp: read again (OCR) where HP is
                               missing or a panel covers the player list

Everything else read by OCR (gold, level, shop, ...) is left as it is.
With --augments-only just the two augment bars are read again (after a
change to src/vision/augments.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
from tqdm import tqdm

from src.vision.arena import arena_signature, is_planning, timer_fill
from src.vision.augments import AugmentReader, expected_counts
from src.vision.items import ItemColumnReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--reread-hp", action="store_true",
                        help="Re-read HP where it is missing or a panel covers the player list")
    parser.add_argument("--augments-only", action="store_true", help="Only read the augment bars again")
    args = parser.parse_args()

    augments, items = AugmentReader(), ItemColumnReader()
    if args.reread_hp:
        from src.vision.ocr_pipeline import hp_panel_open, read_hp
    vod_dir = Path(args.frames).expanduser()

    for states in sorted(Path(args.states).glob("game_*.jsonl")):
        if states.name.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
        missing = reread = 0
        for row in tqdm(rows, desc=states.stem, unit="row", dynamic_ncols=True,
                        smoothing=0, mininterval=1.0):
            if not row.get("in_game"):
                continue
            frame = cv2.imread(str(vod_dir / states.stem / f"frame_{row['frame']:06d}.jpg"))
            if frame is None:
                missing += 1
                continue
            # the stage limits how many icons the bar can hold, so each icon is cut on its own
            augs = augments.read(frame, expected_counts(row.get("stage"), row.get("round_num")))
            row["augments"] = [a.name for a in augs]
            row["augment_tiers"] = [a.tier for a in augs]
            row["away_augments"] = augments.count_away(frame)
            if args.augments_only:
                continue
            row["arena_sig"] = arena_signature(frame)
            row["planning"] = is_planning(frame)
            row["timer_fill"] = timer_fill(frame)
            row["bench_items"] = [n.split("~")[0] if n else None for n in items.read(frame)]
            if args.reread_hp and (row.get("hp") is None or hp_panel_open(frame)):
                row["hp"], row["other_hp"] = read_hp(frame)
                reread += 1
        states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"  {states.name}: {len(rows)} rows, {reread} HP readings re-read, "
              f"{missing} frames missing")


if __name__ == "__main__":
    main()
