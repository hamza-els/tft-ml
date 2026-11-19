"""
Many labelled augment icons from each labelled game.

A game's augments never change once picked, and the bar fills left to right,
so the icon in slot k is the same augment on every own-board frame where the
bar shows at least k+1 icons. Each game Hamza labelled (scripts/label_augments.py)
therefore gives hundreds of examples of its augments, in different scenery,
lighting and compression, instead of one.

Usage:
    python scripts/collect_augment_crops.py                 # every labelled game
    python scripts/collect_augment_crops.py --per-game 80   # frames sampled per game

Per labelled game: up to --per-game own-board frames spread over the game
(away fights excluded: there the left bar is the opponent's), each cut with
AugmentReader at the counts its stage allows. A slot is used only if that frame
shows the slot. Crops spoiled by something passing over the bar are dropped:
per game and slot, only crops that correlate with the slot's medoid crop
(the one most like all the others) at >= --min-agree are kept.

Output (local): ~/tft_data/augment_crops/frames/<vod>__<game>__a<k>__<frame>.png
and index.json {file: augment name} next to them ("+" versions folded, see
GameData.augment_label). Games whose label has "_unknown" in a slot skip it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm

from src.state.game_data import GameData
from src.utils import paths
from src.vision.arena import home_board_rows
from src.vision.augments import AugmentReader, expected_counts

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
OUT = paths.LOCAL / "augment_crops" / "frames"


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(a, b[2:-2, 2:-2], cv2.TM_CCOEFF_NORMED).max())


def keep_agreeing(crops: list[tuple[int, np.ndarray]], min_agree: float) -> list[tuple[int, np.ndarray]]:
    """Crops that correlate with the medoid (the crop most like the others) at >= min_agree."""
    if len(crops) < 3:
        return crops
    sample = crops if len(crops) <= 40 else [crops[i] for i in np.linspace(0, len(crops) - 1, 40).astype(int)]
    medoid = max(sample, key=lambda c: sum(ncc(c[1], o[1]) for o in sample if o is not c))[1]
    return [c for c in crops if ncc(c[1], medoid) >= min_agree]


def frame_dir(vod: str, game: str) -> Path | None:
    for root in (paths.FRAMES, paths.AUG_FRAMES):
        d = root / vod / game
        if d.exists():
            return d
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-game", type=int, default=80, help="Own-board frames sampled per game")
    parser.add_argument("--min-agree", type=float, default=0.6, help="Correlation with the slot's medoid crop")
    args = parser.parse_args()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    reader = AugmentReader()
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index: dict[str, str] = {}
    kept = dropped = 0
    for key in tqdm(sorted(labels), desc="games", unit="game"):
        vod, game = key.split("/")
        names = [GameData.augment_label(n) for n in labels[key]["augments"]]
        states = paths.STATES / vod / f"{game}.jsonl"
        folder = frame_dir(vod, game)
        if not states.exists() or folder is None:
            continue
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        home = home_board_rows(rows)
        # own board, the bar showing something, not a fight on someone's board (both bars up)
        cands = [r for r, h in zip(rows, home) if h and r.get("augments") and not r.get("away_augments")]
        if not cands:
            continue
        picks = [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, min(args.per_game, len(cands))).astype(int))]
        per_slot: dict[int, list[tuple[int, np.ndarray]]] = {}
        for r in picks:
            img = cv2.imread(str(folder / f"frame_{r['frame']:06d}.jpg"))
            if img is None:
                continue
            for k, icon in enumerate(reader.icons(img, expected_counts(r.get("stage"), r.get("round_num")))):
                if k < len(names) and not names[k].startswith("_"):
                    per_slot.setdefault(k, []).append((r["frame"], icon))
        for k, crops in per_slot.items():
            good = keep_agreeing(crops, args.min_agree)
            dropped += len(crops) - len(good)
            for frame, icon in good:
                name = f"{vod}__{game}__a{k}__{frame:06d}.png"
                cv2.imwrite(str(OUT / name), icon)
                index[name] = names[k]
                kept += 1
    index_path.write_text(json.dumps(index, indent=0, sort_keys=True), encoding="utf-8")
    augments = {}
    for n in index.values():
        augments[n] = augments.get(n, 0) + 1
    print(f"{kept} crops kept ({dropped} dropped as not like their slot's others), "
          f"{len(augments)} augments, median {int(np.median(list(augments.values())))} crops each -> {OUT}")


if __name__ == "__main__":
    main()
