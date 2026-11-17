"""
Collect labelled champion pictures automatically, using shop purchases.

When a unit is bought, three things are already known from the extracted
states: which shop card disappeared (so the champion's name, from OCR), that
gold dropped by exactly that champion's cost, and — by comparing the frame
before and after — which bench slot changed. The picture of that bench slot
is saved under the champion's name.

Usage:
    python scripts/harvest_bench_units.py --frames ~/tft_frames_temp/VOD_ID \
        --states data/processed/state_vectors/VOD_ID

Output:
    data/processed/champion_crops/<Champion>/<vod>__<game>__f<frame>_s<slot>.jpg
    data/processed/champion_crops/_empty/...      empty bench slots (negative examples)
    data/raw/review/champion_crops.jpg            overview sheet, one row per champion

Review before training: delete any picture that doesn't show the named
champion standing alone in the slot. Known ways a crop can be wrong:
  - the purchase completed a 2-star, so the bench slot emptied instead
  - the unit went straight to the board (bench full or auto-placed)
  - another unit was dragged across the slot in the same moment
Slots hidden behind a tooltip are ignored when deciding where the unit landed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from src.state.game_data import load_game_data
from src.vision.bench import BenchSlots

OUT_DIR = Path("data/processed/champion_crops")
SHEET = Path("data/raw/review/champion_crops.jpg")
MAX_GAP_FRAMES = 6        # before/after frames at most 3 s apart
MIN_CHANGE = 12.0         # the slot the unit landed in must change at least this much
DOMINANCE = 1.8           # ...and this many times more than any other slot


def find_purchases(rows: list[dict], game) -> list[tuple[dict, dict, str]]:
    """(row before, row after, champion) for each clean single-unit purchase."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if not (a.get("shop_visible") and b.get("shop_visible")):
            continue
        if b["frame"] - a["frame"] > MAX_GAP_FRAMES:
            continue
        if (a.get("stage"), a.get("round_num")) != (b.get("stage"), b.get("round_num")):
            continue
        sa, sb = a.get("shop_slots") or [], b.get("shop_slots") or []
        if len(sa) != 5 or len(sb) != 5:
            continue
        gone = [i for i in range(5) if sa[i] != sb[i]]
        if len(gone) != 1 or sa[gone[0]] is None or sb[gone[0]] is not None:
            continue
        name = sa[gone[0]]
        if name not in game.champions:
            continue
        if a.get("gold") is None or b.get("gold") is None:
            continue
        if a["gold"] - b["gold"] != game.cost(name):
            continue
        out.append((a, b, name))
    return out


def build_sheet(out_dir: Path, per_row: int = 14, tile: tuple[int, int] = (70, 84)) -> np.ndarray | None:
    rows = []
    for folder in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.jpg"))
        if not files:
            continue
        step = max(1, len(files) // per_row)
        label = np.zeros((tile[1], 190, 3), np.uint8)
        cv2.putText(label, folder.name[:16], (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(label, f"{len(files)} crops", (4, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160),
                    1, cv2.LINE_AA)
        tiles = [cv2.resize(cv2.imread(str(f)), tile) for f in files[::step][:per_row]]
        tiles += [np.zeros((tile[1], tile[0], 3), np.uint8)] * (per_row - len(tiles))
        rows.append(np.hstack([label] + tiles))
    return np.vstack(rows) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--fresh", action="store_true", help="Delete existing crops first")
    args = parser.parse_args()

    vod_dir = Path(args.frames).expanduser()
    game = load_game_data()
    bench = BenchSlots()
    if args.fresh and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    saved = rejected = empties = 0
    for states in sorted(Path(args.states).glob("game_*.jsonl")):
        if states.name.endswith(".clean.jsonl"):
            continue
        folder = vod_dir / states.stem
        rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
        rows.sort(key=lambda r: r["frame"])
        prefix = f"{vod_dir.name}__{states.stem}"

        for a, b, name in tqdm(find_purchases(rows, game), desc=states.stem, unit="buy",
                               dynamic_ncols=True, smoothing=0, mininterval=1.0):
            before = cv2.imread(str(folder / f"frame_{a['frame']:06d}.jpg"))
            after = cv2.imread(str(folder / f"frame_{b['frame']:06d}.jpg"))
            if before is None or after is None:
                continue
            changes = bench.slot_changes(before, after)
            # A tooltip over the bench changes many slots at once; leave those slots out.
            hidden = [x or y for x, y in zip(bench.covered(before), bench.covered(after))]
            changes = [0.0 if h else c for c, h in zip(changes, hidden)]
            order = np.argsort(changes)[::-1]
            top, second = changes[order[0]], changes[order[1]]
            if top < MIN_CHANGE or top < DOMINANCE * second:
                rejected += 1
                continue
            slot = int(order[0])
            # A unit that was there before and is gone now left the slot (2-star merge, or moved).
            if bench.has_health_bar(before, slot) and not bench.has_health_bar(after, slot):
                rejected += 1
                continue
            dest = OUT_DIR / name
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / f"{prefix}__f{b['frame']:06d}_s{slot}.jpg"), bench.crop(after, slot))
            saved += 1

        # Empty-slot examples: the bench's right end during stage 1 is almost always empty.
        early = [r for r in rows if r.get("stage") == 1 and r.get("shop_visible")][::6][:12]
        for r in early:
            frame = cv2.imread(str(folder / f"frame_{r['frame']:06d}.jpg"))
            if frame is None:
                continue
            for slot in (6, 7, 8):
                dest = OUT_DIR / "_empty"
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / f"{prefix}__f{r['frame']:06d}_s{slot}.jpg"), bench.crop(frame, slot))
                empties += 1

    champions = [p for p in OUT_DIR.iterdir() if p.is_dir() and p.name != "_empty"]
    print(f"\n{saved} champion crops across {len(champions)} champions "
          f"({rejected} purchases skipped: no single clear slot, or the slot emptied), {empties} empty-slot crops")
    missing = sorted(set(game.champion_names) - {p.name for p in champions})
    print(f"No crops yet for {len(missing)}: {missing}")

    sheet = build_sheet(OUT_DIR)
    if sheet is not None:
        SHEET.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Overview sheet: {SHEET}")


if __name__ == "__main__":
    main()
