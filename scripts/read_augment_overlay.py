"""
Each game's augments (and portal) read from the stream overlay
(src/vision/augment_overlay.py), voted over many late-game frames.

Usage:
    python scripts/read_augment_overlay.py VOD [VOD ...]       # -> ~/tft_data/state_vectors/<VOD>/augment_overlay.json
    python scripts/read_augment_overlay.py --all                # every VOD whose full frames are on disk
    python scripts/read_augment_overlay.py --all --eval         # also score against data/processed/augment_labels.json

Needs the full frames (~/tft_frames_temp/<VOD>): the augment-only copies
(scripts/slim_frames.py) black the overlay out.

Per game: up to --frames frames after the last augment pick (4-2), spread over
the rest of the game; the overlay is the player's own whatever board is on
screen. Readings are voted per position: the length is the longest one at
least 3 frames (and 30% of the readable ones) agree on, then each position
takes its commonest name among those frames.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from src.state.game_data import GameData
from src.utils import paths
from src.vision.augment_overlay import read
from src.vision.ocr_pipeline import _get_reader

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"


def vote(readings: list[list[str]]) -> list[str]:
    readings = [r for r in readings if r]
    if not readings:
        return []
    lengths = Counter(len(r) for r in readings)
    ok = [n for n, k in lengths.items() if k >= max(3, 0.3 * len(readings))]
    m = max(ok) if ok else lengths.most_common(1)[0][0]
    same = [r for r in readings if len(r) == m]
    return [Counter(r[k] for r in same).most_common(1)[0][0] for k in range(m)]


def read_game(states: Path, frames_dir: Path, reader, n_frames: int) -> dict:
    rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                  key=lambda r: r["frame"])
    late = [r for r in rows if r.get("in_game") and (r.get("stage") or 0, r.get("round_num") or 0) > (4, 2)]
    if not late:
        late = [r for r in rows if r.get("in_game")][-n_frames:]
    picks = [late[i] for i in np.unique(np.linspace(0, len(late) - 1, min(n_frames, len(late))).astype(int))] if late else []
    readings, portals = [], Counter()
    for r in picks:
        img = cv2.imread(str(frames_dir / f"frame_{r['frame']:06d}.jpg"))
        if img is None:
            continue
        portal, augs = read(img, reader)
        readings.append(augs)
        if portal:
            portals[portal] += 1
    return {"augments": vote(readings), "portal": portals.most_common(1)[0][0] if portals else None,
            "frames": len(picks), "readable": sum(1 for r in readings if r)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vods", nargs="*")
    parser.add_argument("--all", action="store_true", help="Every VOD whose full frames are on disk")
    parser.add_argument("--frames", type=int, default=25, help="Frames read per game")
    parser.add_argument("--eval", action="store_true", help="Score against Hamza's augment labels")
    args = parser.parse_args()
    vods = args.vods or []
    if args.all:
        vods += sorted(p.name for p in paths.STATES.iterdir() if (paths.FRAMES / p.name).exists())
    reader = _get_reader()
    results = {}
    for vod in dict.fromkeys(vods):
        out = {}
        for states in tqdm(sorted((paths.STATES / vod).glob("game_??.jsonl")), desc=vod, unit="game"):
            frames_dir = paths.FRAMES / vod / states.stem
            if frames_dir.exists():
                out[states.stem] = read_game(states, frames_dir, reader, args.frames)
        (paths.STATES / vod / "augment_overlay.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        results.update({f"{vod}/{g}": v for g, v in out.items()})

    if args.eval:
        labels = json.loads(LABELS.read_text(encoding="utf-8"))
        fold = GameData.augment_label
        base = GameData.augment_base
        games = slots = pos_ok = pos_base = set_ok = same_order = read_games = 0
        misses = Counter()
        for key, lab in labels.items():
            if key not in results:
                continue
            games += 1
            got = results[key]["augments"]
            truth = [n for n in lab["augments"]]
            if got:
                read_games += 1
            for k, t in enumerate(truth):
                if t.startswith("_"):
                    continue
                slots += 1
                g = got[k] if k < len(got) else None
                pos_ok += g is not None and fold(g) == fold(t)
                pos_base += g is not None and base(g) == base(t)
                set_ok += any(base(x) == base(t) for x in got)
                if not (g is not None and base(g) == base(t)):
                    misses[(t, g)] += 1
            tb = [base(t) for t in truth if not t.startswith("_")]
            gb = [base(x) for x in got]
            same_order += bool(gb) and [x for x in gb if x in tb] == [x for x in tb if x in gb] and len(set(gb) & set(tb)) >= 2
        print(f"\n{games} labelled games with full frames, overlay read in {read_games}")
        print(f"  {slots} labelled slots: right name in the right slot {pos_ok} ({pos_ok / slots:.0%}); "
              f"same augment, any tier, right slot {pos_base} ({pos_base / slots:.0%}); "
              f"anywhere in the game's list {set_ok} ({set_ok / slots:.0%})")
        print(f"  games whose shared augments come in the same order as the bar: {same_order} of {read_games}")
        print("  misses (label -> overlay):")
        for (t, g), n in misses.most_common(20):
            print(f"    {n}x {t} -> {g}")


if __name__ == "__main__":
    main()
