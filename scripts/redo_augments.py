"""
Redo augment detection on VODs that already went through the pipeline, then
re-clean, re-encode and re-upload their states. Nothing else is re-read: OCR
values, units and items stay as they are in the state files.

Usage:
    python scripts/redo_augments.py VOD_ID [VOD_ID ...]
    python scripts/redo_augments.py --all          # every finished VOD (~/tft_data/state_vectors/*/.done)

Frames: the full frames (~/tft_frames_temp/VOD) when they are still there,
else the augment-only copies (~/tft_frames_augments/VOD, scripts/slim_frames.py).

Steps per VOD (edit STEPS when the way augments are read changes):
    harvest_augment_icons.py --update-states   new icon templates + augments/augment_tiers per frame
    clean_states.py                            per-game augment decision (and labels from label_augments.py)
    build_dataset.py                           .npz
    upload_states.py                           copy into the repo

Note: `home_board` (own board vs someone else's) comes partly from the augment
bars, but which frames had their units read was fixed when read_units.py ran.
If a change here moves frames into "own board", their units can only be read
from full frames.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import subprocess
import time

from src.data_collection.frame_dedup import frame_folders
from src.utils import paths

ROOT = Path(__file__).resolve().parents[1]
LOG = paths.LOGS / "redo_augments.log"


def frames_for(vid: str) -> Path | None:
    for root in (paths.FRAMES / vid, paths.AUG_FRAMES / vid):
        if root.exists() and frame_folders(str(root)):
            return root
    return None


def steps(vid: str, frames: Path) -> list[tuple[str, list[str]]]:
    states = paths.STATES / vid
    return [
        # read both bars again with every template: own augments and away_augments (harvest doesn't
        # write away_augments, which decides home_board)
        ("augments", ["scripts/backfill_states.py", "--augments-only", "--frames", str(frames), "--states", str(states)]),
        ("clean states", ["scripts/clean_states.py", "--states", str(states)]),
        ("encode dataset", ["scripts/build_dataset.py", "--states", str(states)]),
        ("upload states", ["scripts/upload_states.py", vid]),
    ]


def run(step: str, args: list[str]) -> bool:
    start = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {step}\n")
        log.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"    {step:<16} {status}  ({(time.time() - start) / 60:.0f} min)", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vods", nargs="*")
    parser.add_argument("--all", action="store_true", help="Every VOD with a finished state folder")
    args = parser.parse_args()
    vods = args.vods or []
    if args.all:
        vods += sorted(p.parent.name for p in paths.STATES.glob("*/.done"))
    if not vods:
        parser.error("give VOD IDs or --all")
    LOG.parent.mkdir(parents=True, exist_ok=True)

    failed = []
    for vid in dict.fromkeys(vods):
        frames = frames_for(vid)
        if frames is None or not (paths.STATES / vid).exists():
            print(f"  {vid}: no frames or no states - skipped")
            failed.append(vid)
            continue
        print(f"  {vid}  (frames: {frames})", flush=True)
        if not all(run(step, a) for step, a in steps(vid, frames)):
            failed.append(vid)
    print(f"\nDone. {len(vods) - len(failed)}/{len(vods)} VODs redone. Log: {LOG}")
    for vid in failed:
        print(f"  failed: {vid}")


if __name__ == "__main__":
    main()
