"""
Read the player's board and bench units into existing state files.

Usage:
    python scripts/read_units.py --frames ~/tft_frames_temp/VOD_ID \
        --states data/processed/state_vectors/VOD_ID

For every own-board planning frame (the only frames where the units on
screen are the player's own and standing on their hexes), adds

    units_raw: [{"at": "h25", "name": "Ashe", "p": 0.97, "star": 2, "items": [...]}, ...]
    board_counter: [units, team size]   the game's own counter (src/vision/board_counter.py)

to the row in game_XX.jsonl (see src/vision/unit_reader.py). Rows that were
not read have no units_raw; units_raw = [] means read and empty.

These are single-frame readings. scripts/clean_states.py turns them into the
cleaned `board` / `bench` of every frame (noise smoothed out over time,
checked against the player's level, carried through combat).

Re-run after retraining the classifier or rebuilding the item templates.
Don't run it on a VOD that another job is still writing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
from tqdm import tqdm

from src.vision.arena import home_board_rows, is_planning, planning_rows
from src.vision.unit_reader import UnitReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--counter-only", action="store_true",
                        help="Only (re)read the board counter on frames whose units were already read")
    args = parser.parse_args()

    vod_dir = Path(args.frames).expanduser()
    reader = UnitReader()   # loads the classifier even for --counter-only (a few seconds)

    for states in sorted(Path(args.states).glob("game_*.jsonl")):
        if states.name.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
        in_game = sorted((r for r in rows if r.get("in_game")), key=lambda r: r["frame"])
        if not any(r.get("timer_fill") is not None for r in in_game):
            print(f"  {states.name}: no timer readings - run scripts/backfill_states.py first")
            continue
        wanted = {r["frame"] for r, home, plan in zip(in_game, home_board_rows(in_game), planning_rows(in_game))
                  if home and plan}

        read = missing = 0
        if args.counter_only:
            for row in tqdm(rows, desc=states.stem, unit="row", dynamic_ncols=True, smoothing=0, mininterval=1.0):
                if row.get("units_raw") is None:
                    continue
                frame = cv2.imread(str(vod_dir / states.stem / f"frame_{row['frame']:06d}.jpg"))
                if frame is None:
                    missing += 1
                    continue
                row["board_counter"] = reader.read_counter(frame)
                read += 1
            states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            print(f"  {states.name}: counter read on {read} frames, {missing} frames missing")
            continue
        for row in tqdm(rows, desc=states.stem, unit="row", dynamic_ncols=True, smoothing=0, mininterval=1.0):
            row.pop("units_raw", None)
            row.pop("board_counter", None)
            if row["frame"] not in wanted:
                continue
            frame = cv2.imread(str(vod_dir / states.stem / f"frame_{row['frame']:06d}.jpg"))
            if frame is None:
                missing += 1
                continue
            # Re-checked on the frame: states extracted before the "Combat"
            # banner check existed still count those frames as planning.
            if not is_planning(frame):
                row["planning"] = False
                continue
            row["units_raw"] = reader.read(frame)
            row["board_counter"] = reader.read_counter(frame)
            read += 1
        states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"  {states.name}: {len(rows)} rows, units read on {read}, {missing} frames missing")


if __name__ == "__main__":
    main()
