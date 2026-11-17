"""
Clean extracted state sequences and print a quality report per game.

Usage:
    python scripts/clean_states.py --states data/processed/state_vectors/VOD_ID

Input:   <states>/game_XX.jsonl          (from scripts/extract_states.py)
Output:  <states>/game_XX.clean.jsonl    (misreads fixed, gaps forward-filled)

The report shows how often each value was read and how many readings broke
the game's rules — a quick check that the OCR regions are still calibrated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.state.sequence import clean_game, quality_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.states).glob("game_*.jsonl")
                   if not p.name.endswith(".clean.jsonl"))
    if not files:
        sys.exit(f"No game_*.jsonl files in {args.states}")

    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        rows = [r for r in rows if r.get("in_game")]
        if not rows:
            print(f"{path.name}: no in-game frames")
            continue
        cleaned = clean_game(rows)
        out = path.with_name(path.stem + ".clean.jsonl")
        out.write_text("".join(json.dumps(r) + "\n" for r in cleaned), encoding="utf-8")

        rep = quality_report(rows, cleaned)
        print(f"\n{path.name}: {rep['frames']} frames, bottom bar visible in {rep['bottom_bar_visible']}")
        print(f"  start {rep['first']}")
        print(f"  end   {rep['last']}")
        for field, stats in rep["fields"].items():
            rate = "  n/a" if stats["read_rate"] is None else f"{stats['read_rate'] * 100:5.1f}%"
            print(f"  {field:<11} read {rate}   corrected {stats['misreads_corrected']}")


if __name__ == "__main__":
    main()
