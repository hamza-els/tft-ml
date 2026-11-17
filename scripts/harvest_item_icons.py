"""
Collect item icons from the item column of real footage, and (optionally)
write the unequipped items into already-extracted state files.

Usage:
    python scripts/harvest_item_icons.py --frames ~/tft_frames_temp/VOD_ID
    python scripts/harvest_item_icons.py --frames ~/tft_frames_temp/VOD_ID \
        --update-states data/processed/state_vectors/VOD_ID

Every icon seen in several frames is saved to assets/templates/items_seen/:
  - under the official item name when the official art matches strongly
    (a second look-alike of the same item gets a "~2" suffix), or
  - as item_XXX.png otherwise. Rename the file to name the item, or delete it
    if it is not an item (a cursor or overlay caught in a slot).

Requiring several sightings keeps one-off junk (a dragged unit, a tooltip)
from becoming a template.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import cv2
from tqdm import tqdm

from src.data_collection.frame_dedup import frame_folders
from src.data_collection.frame_game_splitter import frame_number
from src.vision.hud import HudDetector
from src.vision.items import ItemColumnReader

MIN_SIGHTINGS = 5


def base_name(name: str) -> str:
    """'Giant's Belt~2' -> 'Giant's Belt' (look-alike copies of one item)."""
    return name.split("~")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--update-states", help="Folder of game_XX.jsonl files to fill bench_items into")
    args = parser.parse_args()

    reader = ItemColumnReader()
    hud = HudDetector()
    reader.seen_dir.mkdir(parents=True, exist_ok=True)
    start_count = len(reader.seen.names)
    pending: list[list] = []      # [icon, sightings]

    def save_new(icon) -> str:
        official, score = reader.official.best(icon)
        if score >= reader.official_threshold:
            name, n = official, 2
            while name in reader.seen.names:
                name, n = f"{official}~{n}", n + 1
        else:
            used = [int(x[5:]) for x in reader.seen.names if x.startswith("item_") and x[5:].isdigit()]
            name = f"item_{max(used, default=0) + 1:03d}"
        cv2.imwrite(str(reader.seen_dir / f"{name}.png"), icon)
        reader.seen.add(name, icon)
        return name

    for folder in frame_folders(str(Path(args.frames).expanduser())):
        paths = sorted(folder.glob("frame_*.jpg"))
        for path in tqdm(paths, desc=folder.name, unit="frame", dynamic_ncols=True,
                         smoothing=0, mininterval=1.0):
            frame = cv2.imread(str(path))
            if frame is None or not hud.detect(frame).stage_visible:
                continue
            for icon in reader.slot_icons(frame):
                if reader.seen.best(icon)[1] >= reader.seen_threshold:
                    continue
                for k, cand in enumerate(pending):
                    if float(reader.seen.prepare(cand[0]) @ reader.seen.prepare(icon)) >= reader.seen_threshold:
                        cand[1] += 1
                        if cand[1] >= MIN_SIGHTINGS:
                            save_new(cand[0])
                            del pending[k]   # list.remove would compare arrays
                        break
                else:
                    pending.append([icon.copy(), 1])

        if args.update_states:
            states = Path(args.update_states) / f"{folder.name}.jsonl"
            if not states.exists():
                print(f"  {states} not found - skipping update")
                continue
            rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
            for row in tqdm(rows, desc=f"{folder.name} states", unit="row", dynamic_ncols=True,
                            smoothing=0, mininterval=1.0):
                frame = cv2.imread(str(folder / f"frame_{row['frame']:06d}.jpg"))
                if frame is None or not row.get("in_game"):
                    continue
                row["bench_items"] = [base_name(n) if n else None for n in reader.read(frame)]
            states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    named = sum(1 for n in reader.seen.names if not n.startswith("item_"))
    print(f"\n{len(reader.seen.names) - start_count} new icon(s); {len(reader.seen.names)} total in "
          f"{reader.seen_dir} ({named} named from official art, "
          f"{len(reader.seen.names) - named} unnamed)")


if __name__ == "__main__":
    main()
