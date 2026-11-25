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
--jobs N reads N games at once (one worker process each; games are independent).
Don't run it on a VOD that another job is still writing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm

from src.utils.config import vision_cfg
from src.vision.arena import combat_vetoed, home_board_rows, is_planning, planning_rows, pve_round


_READER = None


def _reader():
    """
    One UnitReader per process (models load once per worker). Imported here so that with
    --jobs the parent, which only hands out games, doesn't load torch (~1.5 GB of commit).
    """
    global _READER
    if _READER is None:
        from src.vision.unit_reader import UnitReader
        _READER = UnitReader()
    return _READER


_FRAME_CLF = None


def _frame_clf():
    """The planning-or-combat frame classifier, if trained (one per process)."""
    global _FRAME_CLF
    if _FRAME_CLF is None:
        from src.vision.frame_classifier import FrameClassifier
        _FRAME_CLF = FrameClassifier() if FrameClassifier.available() else False
    return _FRAME_CLF or None


def process_game(states_path: str, vod_dir_path: str, light: bool, progress: bool = True) -> str:
    """Read (or with light, refresh) the units of one game's state file; returns a summary line."""
    states, vod_dir = Path(states_path), Path(vod_dir_path)
    reader = _reader()
    rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
    in_game = sorted((r for r in rows if r.get("in_game")), key=lambda r: r["frame"])
    if not any(r.get("timer_fill") is not None for r in in_game):
        return f"  {states.name}: no timer readings - run scripts/backfill_states.py first"
    wanted = {r["frame"] for r, home, plan in zip(in_game, home_board_rows(in_game),
                                                  planning_rows(in_game, pve_candidates=True))
              if home and plan}
    bar = tqdm(rows, desc=states.stem, unit="row", dynamic_ncols=True, smoothing=0, mininterval=1.0,
               disable=not progress)
    read = missing = 0
    if light:
        covered_n = 0
        for row in bar:
            if row.get("units_raw") is None:
                continue
            frame = cv2.imread(str(vod_dir / states.stem / f"frame_{row['frame']:06d}.jpg"))
            if frame is None:
                missing += 1
                continue
            row["board_counter"] = reader.read_counter(frame)
            covered = set(reader.covered(frame))
            kept = [u for u in row["units_raw"] if u["at"] not in covered and not u.get("covered")]
            row["units_raw"] = kept + [{"at": at, "covered": True} for at in sorted(covered)]
            covered_n += len(covered)
            read += 1
        states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return (f"  {states.name}: counter and tooltips read on {read} frames "
                f"({covered_n} positions under a tooltip), {missing} frames missing")
    for row in bar:
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
        pve = pve_round(row.get("stage"), row.get("round_num"))
        far_half_clear = is_planning(frame)                  # the usual rule (no bars on the far half)
        if not (far_half_clear or (pve and is_planning(frame, pve=True))):
            row["planning"] = False
            continue
        row["planning"] = far_half_clear
        counter = reader.read_counter(frame)
        if pve and not far_half_clear and not (counter and counter[1] is not None):
            # PvE round: only frames showing the planning-only "units / team size" counter
            # (arena.planning_rows); in a fight the monsters' bars don't tell it apart
            row["board_counter"] = counter
            continue
        # The frame classifier vetoes a fight that passed the rules for an instant, unless the
        # planning-only "units / team size" counter is up (arena.combat_vetoed, also read by
        # arena.planning_rows); a vetoed frame's units are not read.
        clf = _frame_clf()
        if clf is not None:
            row["combat_p"] = round(clf.combat_p([frame])[0], 3)
            if combat_vetoed(row["combat_p"], counter):
                row["board_counter"] = counter
                continue
        row["units_raw"] = reader.read(frame)
        row["board_counter"] = counter
        read += 1
    states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return f"  {states.name}: {len(rows)} rows, units read on {read}, {missing} frames missing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--light", "--counter-only", dest="light", action="store_true",
                        help="On frames whose units were already read, only (re)read the board counter "
                             "and which positions are under a tooltip (no re-classifying)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Games read in parallel, one worker process each (models load per worker)")
    args = parser.parse_args()

    vod_dir = str(Path(args.frames).expanduser())
    games = [str(p) for p in sorted(Path(args.states).glob("game_*.jsonl")) if not p.name.endswith(".clean.jsonl")]
    if args.jobs <= 1:
        for g in games:
            print(process_game(g, vod_dir, args.light))
        return
    # Whole games per worker: each file is read and written by one process only.
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process_game, g, vod_dir, args.light, False) for g in games]
        for f in futures:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
