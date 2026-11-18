"""
Draw every configured region on a frame and save the picture, to check at a
glance that the regions line up with this stream's HUD.

Usage:
    python scripts/preview_regions.py path/to/frame.jpg
    python scripts/preview_regions.py path/to/frame.jpg --out preview.jpg

Colours:
    magenta — OCR regions (gold, level, xp, streak, stage, HP strip)
    orange  — shop name slots
    yellow  — dedup watch regions
    green   — HUD icon search windows (thick box = icon found)
    cyan    — augment bar (full three-icon extent)
    white   — item column slots and bench slots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import cv2

from src.data_collection.frame_dedup import WATCH_REGIONS
from src.utils.config import vision_cfg
from src.vision.augments import AugmentReader
from src.vision.bench import BenchSlots
from src.vision.hud import HudDetector, shift
from src.vision.items import ItemColumnReader

MAGENTA, ORANGE, YELLOW, GREEN = (255, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0)
CYAN, WHITE = (255, 255, 0), (255, 255, 255)


def _box(img, region, colour, label=None, thickness=2):
    x, y, w, h = region
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, thickness)
    if label:
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                    cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame")
    parser.add_argument("--out", default=None, help="Output path (default: <frame>_regions.jpg)")
    args = parser.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        sys.exit(f"Could not read image: {args.frame}")

    ocr = vision_cfg["ocr"]
    hud_cfg = vision_cfg["hud"]
    detector = HudDetector()
    hud = detector.detect(img)

    dx = hud.bottom_dx   # bottom bar position in this frame's layout
    for name, region in WATCH_REGIONS.items():
        _box(img, shift(region, dx) if name in ("gold", "shop") else region, YELLOW, f"dedup:{name}", 1)
    for name, region in ocr["regions"].items():
        _box(img, shift(region, dx), MAGENTA, name)
    _box(img, shift(ocr["streak_icon"], dx), MAGENTA)
    _box(img, ocr["hp_strip"], MAGENTA, "hp")
    for slot in ocr["shop_slots"]:
        _box(img, shift(slot, dx), ORANGE)

    augments = AugmentReader()
    n_augments = augments.count(img)
    _box(img, augments.region, CYAN, f"augments ({n_augments})", 1)
    n_away = augments.count_away(img)
    _box(img, augments.away_region, CYAN, f"other player's augments ({n_away})", 1)

    items, bench = ItemColumnReader(), BenchSlots()
    for i in range(items.slots):
        _box(img, items.slot_region(i), WHITE, None, 1)
    for i in range(bench.slots):
        _box(img, bench.region(i), WHITE, f"b{i}", 1)

    _box(img, hud_cfg["stage_icon_search"], GREEN, "stage icon search", 1)
    _box(img, hud_cfg["gold_icon_search"], GREEN, None, 1)
    if hud.stage_visible:
        ix, iy = hud.stage_icon_xy
        th, tw = detector.stage_icon.shape[:2]
        _box(img, (ix, iy, tw, th), GREEN, None, 3)
        dx, dy, w, h = ocr["stage_text_offset"]
        _box(img, (ix + dx, max(0, iy + dy), w, h), MAGENTA, "stage")

    out = args.out or str(Path(args.frame).with_name(Path(args.frame).stem + "_regions.jpg"))
    cv2.imwrite(out, img)
    print(f"stage bar found: {hud.stage_visible}  (stage-1 layout: {hud.stage_one_layout})")
    print(f"bottom bar found: {hud.shop_visible}  (shifted {hud.bottom_dx} px from calibration)")
    print(f"augments shown: {n_augments}")
    print(f"other player's augments shown: {n_away}"
          + ("  -> not the player's own board" if n_away else ""))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
