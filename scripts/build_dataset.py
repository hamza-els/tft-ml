"""
Encode cleaned state sequences into model-ready arrays, one file per game.

Usage:
    python scripts/build_dataset.py --states ~/tft_data/state_vectors/VOD_ID
    python scripts/build_dataset.py --describe        # print the vector layout only

Input:   <states>/game_XX.clean.jsonl        (from scripts/clean_states.py)
Output:  ~/tft_data/datasets/states/<VOD_ID>/game_XX.npz with
           X              float32 [frames, features]   encoded states
           frame          int     [frames]             source frame numbers
           time_sec       float   [frames]             seconds into the VOD
           feature_names  str     [features]
           tok_int        int16   [frames, 38, fields]   per-unit token table (src/state/tokens.py)
           tok_float      float32 [frames, 38, fields]
           tok_mask       bool    [frames, 38]
           placement      int                            final placement label (-1 unknown)
         and ~/tft_data/datasets/states/vocab.json (IDs -> names, field names)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from src.state.encoder import StateEncoder
from src.state.tokens import TokenEncoder, placement
from src.utils import paths

OUT_DIR = paths.DATASETS     # local disk (src/utils/paths.py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", help="Folder of game_XX.clean.jsonl files")
    parser.add_argument("--describe", action="store_true", help="Print the vector layout and exit")
    args = parser.parse_args()

    encoder = StateEncoder()
    tokens = TokenEncoder(encoder.game)
    print(f"State vector: {encoder.size} features  (set {encoder.game.set_name}, "
          f"{len(encoder.game.champion_names)} champions, {len(encoder.game.trait_names)} traits)")
    for name, start, length in encoder.describe():
        print(f"  {start:4d}-{start + length - 1:<4d} {name:<18} {length}")
    print("Token table (src/state/tokens.py):")
    for line in tokens.describe():
        print(f"  {line}")
    if args.describe:
        return
    if not args.states:
        sys.exit("--states is required unless --describe is given")

    states = Path(args.states)
    files = sorted(states.glob("game_*.clean.jsonl"))
    if not files:
        sys.exit(f"No game_*.clean.jsonl in {states} - run scripts/clean_states.py first")

    out_dir = OUT_DIR / states.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # IDs are fixed by the set data; written next to the datasets for decoding.
    (OUT_DIR / "vocab.json").write_text(json.dumps(tokens.vocab(), indent=1), encoding="utf-8")
    total = 0
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        X = encoder.encode_game(rows)
        tok_int, tok_float, tok_mask = tokens.encode_game(rows)
        out = out_dir / (path.name.replace(".clean.jsonl", ".npz"))
        np.savez_compressed(
            out, X=X,
            frame=np.array([r["frame"] for r in rows]),
            time_sec=np.array([r["time_sec"] for r in rows]),
            feature_names=np.array(encoder.feature_names),
            tok_int=tok_int, tok_float=tok_float, tok_mask=tok_mask,
            placement=np.array(placement(rows)),
        )
        total += len(rows)
        print(f"  {path.name}: {X.shape} -> {out}")
    print(f"{total} states in {len(files)} game(s)")


if __name__ == "__main__":
    main()
