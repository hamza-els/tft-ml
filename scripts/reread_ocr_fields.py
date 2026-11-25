"""
Read gold, XP and/or HP (own and lobby) again in existing state files, leaving every other field
(units, augments, items, planning ...) as it is. For fixes to those readers without a whole
re-read, which would also drop the unit readings (units_raw) and need read_units.py again.

Usage:
    python scripts/reread_ocr_fields.py --frames ~/tft_frames_temp/VOD --states ~/tft_data/state_vectors/VOD
    python scripts/reread_ocr_fields.py ... --fields gold xp     # not HP
    python scripts/reread_ocr_fields.py ... --games game_03 game_05

Same code path as extract_states.py: one OcrCache per game (gold re-read only when its text
pixels change, HP read on each round's first frames), the HUD found per frame. Written for the
2026-11-25 fixes: gold cut from the gold icon found on the frame (the old fixed box cut the left
edge of the first of two digits: 4 -> 1, 6 -> 5, 8 -> 3), and lobby rows the whole-strip OCR
missed read one by one. Then clean_states.py and build_dataset.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import cv2

FIELDS = ("gold", "xp", "hp")


def reread_game(frames: Path, states: Path, fields: set[str]) -> str:
    from src.vision.ocr_pipeline import OcrCache, _ALLOWLIST, _parse_number, _read_text, field_region, parse_xp
    from src.vision.hud import HudDetector
    from src.vision.text_bank import text_bank

    rows = [json.loads(line) for line in states.read_text(encoding="utf-8").splitlines() if line]
    hud_det, cache = HudDetector(), OcrCache()
    changed, missing, t0 = {f: 0 for f in FIELDS}, 0, time.time()
    for row in rows:
        if not row.get("in_game"):
            continue
        frame = cv2.imread(str(frames / f"frame_{row['frame']:06d}.jpg"))
        if frame is None:
            missing += 1
            continue
        hud = hud_det.detect(frame)
        if "gold" in fields:
            gold = None
            if hud.shop_visible:
                region = field_region("gold", hud)

                def look(region=region):
                    bank = text_bank()
                    if bank is not None:
                        v = bank.lookup("gold", cv2.cvtColor(frame[region[1]:region[1] + region[3],
                                                                   region[0]:region[0] + region[2]], cv2.COLOR_BGR2GRAY))
                        if v:
                            return v
                    return _read_text(frame, region, _ALLOWLIST["default"])
                gold = _parse_number(cache.get("gold", frame, region, look))
            changed["gold"] += gold != row.get("gold")
            row["gold"] = gold
        if "xp" in fields:
            xp = (None, None)
            if hud.shop_visible:
                region = field_region("xp", hud)

                def look_xp(region=region):
                    bank = text_bank()
                    if bank is not None:
                        v = bank.lookup("xp", cv2.cvtColor(frame[region[1]:region[1] + region[3],
                                                                 region[0]:region[0] + region[2]], cv2.COLOR_BGR2GRAY))
                        if v:
                            return v
                    return _read_text(frame, region, _ALLOWLIST["xp"])
                xp = parse_xp(cache.get("xp", frame, region, look_xp)) or (None, None)
            xp = tuple(None if v is None else float(v) for v in xp)
            changed["xp"] += xp != (row.get("xp_current"), row.get("xp_needed"))
            row["xp_current"], row["xp_needed"] = xp
        if "hp" in fields and hud.stage_visible:
            hp, others = cache.read_hp(frame, (row.get("stage"), row.get("round_num")))
            changed["hp"] += (hp, others) != (row.get("hp"), row.get("other_hp"))
            row["hp"], row["other_hp"] = hp, others
    tmp = states.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for attempt in range(10):                      # Drive / AV can hold the file a moment
        try:
            tmp.replace(states)
            break
        except PermissionError:
            time.sleep(2)
    return (f"{states.name}: {len(rows)} rows, changed " + ", ".join(f"{f} {changed[f]}" for f in sorted(fields))
            + f", {missing} frames missing, {time.time() - t0:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="VOD frame folder containing game_XX/")
    parser.add_argument("--states", required=True, help="Folder of game_XX.jsonl files")
    parser.add_argument("--fields", nargs="+", default=list(FIELDS), choices=FIELDS)
    parser.add_argument("--games", nargs="*", help="Only these games (default: all)")
    args = parser.parse_args()
    frames, states = Path(args.frames).expanduser(), Path(args.states).expanduser()
    for path in sorted(states.glob("game_??.jsonl")):
        if args.games and path.stem not in args.games:
            continue
        if not (frames / path.stem).is_dir():
            print(f"{path.name}: no frames, skipped")
            continue
        print(reread_game(frames / path.stem, path, set(args.fields)), flush=True)


if __name__ == "__main__":
    main()
