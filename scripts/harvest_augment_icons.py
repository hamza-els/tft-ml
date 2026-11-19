"""
Collect augment icons from footage, and (optionally) write the augments into
already-extracted state files without re-running OCR.

Usage:
    # Save every new icon as assets/templates/augments/aug_XXX.png
    python scripts/harvest_augment_icons.py --frames ~/tft_frames_temp/VOD_ID

    # Also fill "augments" / "augment_tiers" in existing state files
    python scripts/harvest_augment_icons.py --frames ~/tft_frames_temp/VOD_ID \
        --update-states data/processed/state_vectors/VOD_ID

An icon is only saved once it has been seen in several frames, so a tooltip or
a unit passing over the bar doesn't become a template. To name an augment,
rename its file (aug_007.png -> Pandoras_Items.png); states extracted after
that use the new name.
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
from src.vision.augments import AugmentReader, expected_counts

MIN_SIGHTINGS = 5


def _next_id(reader: AugmentReader) -> int:
    used = [int(n[4:]) for n in reader.templates if n.startswith("aug_") and n[4:].isdigit()]
    return max(used, default=0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--update-states", help="Folder of game_XX.jsonl files to fill augments into")
    args = parser.parse_args()

    reader = AugmentReader()
    reader.template_dir.mkdir(parents=True, exist_ok=True)
    start_count = len(reader.templates)
    next_id = _next_id(reader)

    for folder in frame_folders(str(Path(args.frames).expanduser())):
        pending: list[list] = []     # [icon, sightings] not yet promoted to templates
        per_frame: dict[int, list[tuple[str | None, str]]] = {}
        # Stage per frame from the state file, if there is one: it limits how many icons
        # the bar can hold, so each icon is cut on its own (AugmentReader, expected_counts).
        allowed: dict[int, set[int] | None] = {}
        states = Path(args.update_states) / f"{folder.name}.jsonl" if args.update_states else None
        if states and states.exists():
            for line in states.read_text(encoding="utf-8").splitlines():
                if line:
                    r = json.loads(line)
                    allowed[r["frame"]] = expected_counts(r.get("stage"), r.get("round_num"))

        for path in tqdm(sorted(folder.glob("frame_*.jpg")), desc=folder.name, unit="frame",
                         dynamic_ncols=True, smoothing=0, mininterval=1.0):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            found = []
            for aug in reader.read(frame, allowed.get(frame_number(path))):
                name = aug.name
                if name is None:
                    for k, cand in enumerate(pending):
                        score = cv2.matchTemplate(aug.icon, cand[0][2:-2, 2:-2],
                                                  cv2.TM_CCOEFF_NORMED).max()
                        if score >= reader.match_threshold:
                            cand[1] += 1
                            if cand[1] >= MIN_SIGHTINGS:
                                name = f"aug_{next_id:03d}"
                                next_id += 1
                                cv2.imwrite(str(reader.template_dir / f"{name}.png"), cand[0])
                                reader.templates[name] = cand[0]
                                del pending[k]   # list.remove would compare arrays
                            break
                    else:
                        pending.append([aug.icon, 1])
                found.append((name, aug.tier))
            per_frame[frame_number(path)] = found

        if args.update_states:
            states = Path(args.update_states) / f"{folder.name}.jsonl"
            if not states.exists():
                print(f"  {states} not found - skipping update")
                continue
            rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
            for row in rows:
                found = per_frame.get(row["frame"], [])
                # Icons seen before their template existed were read as None; match again now.
                row["augments"] = [n for n, _ in found]
                row["augment_tiers"] = [t for _, t in found]
            # second pass for names that were still pending when first seen
            for row in rows:
                if None in row["augments"]:
                    path = folder / f"frame_{row['frame']:06d}.jpg"
                    frame = cv2.imread(str(path))
                    if frame is not None:
                        augs = reader.read(frame, allowed.get(row["frame"]))
                        row["augments"] = [a.name for a in augs]
                        row["augment_tiers"] = [a.tier for a in augs]
            states.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    print(f"\n{len(reader.templates) - start_count} new icon(s), "
          f"{len(reader.templates)} total in {reader.template_dir}")


if __name__ == "__main__":
    main()
