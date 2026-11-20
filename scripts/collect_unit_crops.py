"""
Cut out every unit standing on the player's board and bench, ready to be
labelled with scripts/label_units.py.

Usage:
    python scripts/collect_unit_crops.py --frames ~/tft_frames_temp/VOD_ID \
        --states data/processed/state_vectors/VOD_ID

Which frames are used:
  - only the player's own board (not scouting, not fighting away)
  - only the planning phase (units stand still on their hexes; combat adds
    nothing because the board doesn't change during a fight). Planning is
    told from the round timer bar: frames before it refills in a round.
  - a couple of frames per round, since the board changes slowly

In each chosen frame a window is cut around every board hex and bench slot
that has a unit's health bar over it (src/vision/unit_reader.occupied). A crop that looks the same as the last
one saved for that hex/slot is skipped, so a unit that stands still for ten
rounds is labelled once, not ten times.

Output:
    ~/tft_data/unit_crops/images/<vod>__<game>__f<frame>__h<row><col>.jpg   (board hex; local disk)
    ~/tft_data/unit_crops/images/<vod>__<game>__f<frame>__s<slot>.jpg       (bench slot)
    data/processed/unit_crops/index.json     crop -> source frame and window, for the labeler (repo)

Requires arena_sig / planning in the state files (new extractions have them;
older ones need scripts/backfill_states.py first).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.arena import home_board_rows, is_planning, planning_rows
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.tooltip import panel_in_crop
from src.vision.unit_reader import occupied

OUT_DIR = paths.CROP_DIR            # index.json (in the repo)
FRAMES_PER_ROUND = 2
CANDIDATES_PER_ROUND = 8   # own-board frames checked per round to find planning-phase ones
SAME_UNIT = 0.90           # crops more alike than this are the same unit standing still


def _fingerprint(crop: np.ndarray) -> np.ndarray:
    small = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    small -= small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm > 0 else small


def candidate_frames(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Own-board planning frames per round, thinned to a handful spread across the round."""
    in_game = [r for r in rows if r.get("in_game")]
    in_game.sort(key=lambda r: r["frame"])
    home, planning = home_board_rows(in_game), planning_rows(in_game)
    by_round: dict[tuple, list[dict]] = {}
    for r, at_home, plan in zip(in_game, home, planning):
        if at_home and plan and r.get("stage") is not None:
            by_round.setdefault((r["stage"], r.get("round_num")), []).append(r)
    for key, frames in by_round.items():
        idx = np.unique(np.linspace(0, len(frames) - 1, CANDIDATES_PER_ROUND).round().astype(int))
        by_round[key] = [frames[i] for i in idx]
    return by_round


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    args = parser.parse_args()

    vod_dir = Path(args.frames).expanduser()
    board, bench = BoardHexes(), BenchSlots()
    images = paths.CROP_IMAGES         # local disk
    images.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    before = set(index)

    total = skipped = 0
    for states in sorted(Path(args.states).glob("game_*.jsonl")):
        if states.name.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
        if not any(r.get("timer_fill") is not None for r in rows):
            print(f"  {states.name}: no timer readings - run scripts/backfill_states.py first")
            continue
        prefix = f"{vod_dir.name}__{states.stem}"
        last: dict[str, np.ndarray] = {}   # window id -> fingerprint of the last crop saved there
        saved = 0

        # Planning is checked on the frame itself, latest first, so each round
        # contributes its final board plus an earlier look.
        chosen = []
        for frames in candidate_frames(rows).values():
            kept = 0
            for row in reversed(frames):
                frame_path = vod_dir / states.stem / f"frame_{row['frame']:06d}.jpg"
                frame = cv2.imread(str(frame_path))
                vetoed = (row.get("combat_p") or 0.0) >= float(vision_cfg["arena"]["combat_veto"])
                if frame is not None and is_planning(frame) and not vetoed:
                    chosen.append((row, frame_path, frame))
                    kept += 1
                    if kept == FRAMES_PER_ROUND:
                        break
        chosen.sort(key=lambda item: item[0]["frame"])

        for row, frame_path, frame in tqdm(chosen, desc=states.stem, unit="frame",
                                           dynamic_ncols=True, smoothing=0, mininterval=1.0):

            for wid, (x, y, w, h), _ in occupied(frame, board, bench):
                crop = frame[max(0, y):y + h, max(0, x):x + w]
                if panel_in_crop(crop):          # a tooltip over the unit: nothing to label
                    skipped += 1
                    continue
                fp = _fingerprint(crop)
                if wid in last and float(fp @ last[wid]) >= SAME_UNIT:
                    skipped += 1
                    continue
                last[wid] = fp
                name = f"{prefix}__f{row['frame']:06d}__{wid}.jpg"
                cv2.imwrite(str(images / name), crop)
                index[name] = {"frame_path": str(frame_path), "window": [x, y, w, h],
                               "stage": row.get("stage"), "round": row.get("round_num"),
                               "where": "bench" if wid.startswith("s") else "board"}
                saved += 1
        total += saved
        print(f"  {states.stem}: {saved} crops")

    # Merge into the file as it is now: another script (recut_unit_crops.py)
    # may have changed other entries while this one ran.
    added = {k: v for k, v in index.items() if k not in before}
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    index.update(added)
    index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
    print(f"\n{total} new crops ({skipped} skipped as unchanged) -> {images}")
    print(f"{len(index)} crops in total. Label them with: python scripts/label_units.py")


if __name__ == "__main__":
    main()
