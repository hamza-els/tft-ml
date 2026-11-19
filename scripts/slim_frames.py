"""
Keep only the augment bars of a VOD's in-game frames, so augment detection can
be redone after the full frames are deleted (they take ~6 GB per VOD; these
copies ~0.3 GB).

Each frame is written at full size (1920x1080, same name, same game_XX folder)
with everything outside two windows blacked out: the player's own augment bar
and the bar drawn on someone else's board. So scripts that read the augment
bars at their usual coordinates (harvest_augment_icons.py, label_augments.py,
AugmentReader) work on these copies unchanged. Nothing else on them is usable:
don't run state extraction or unit reading on them.

Usage:
    python scripts/slim_frames.py VOD_ID [VOD_ID ...]                # write ~/tft_frames_augments/VOD_ID
    python scripts/slim_frames.py VOD_ID --delete-full               # and then delete ~/tft_frames_temp/VOD_ID

The windows (WINDOWS below, also saved as windows.json in each copy) leave
~55 px of room on each side of a four-icon bar and ~30 px above and below it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from src.data_collection.frame_dedup import frame_folders
from src.utils import paths

# (x0, y0, x1, y1) in 1920x1080 frames
WINDOWS = {
    "own": (380, 195, 640, 300),       # own bar: centre x 510, y 228-266, up to 4 x 37.5 px
    "away": (1260, 235, 1520, 340),    # someone else's bar: centre x 1390, y 268-306
}
QUALITY = 95


def slim(src: Path, dst: Path):
    if dst.exists():
        return
    img = cv2.imread(str(src))
    if img is None:
        return
    out = np.zeros_like(img)
    for x0, y0, x1, y1 in WINDOWS.values():
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    tmp = dst.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), out, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    tmp.replace(dst)


def slim_vod(vid: str, delete_full: bool) -> bool:
    src_root, dst_root = paths.FRAMES / vid, paths.AUG_FRAMES / vid
    folders = frame_folders(str(src_root)) if src_root.exists() else []
    if not folders:
        print(f"  {vid}: no game frames under {src_root}")
        return False
    n_src = n_dst = 0
    for folder in folders:
        out = dst_root / folder.relative_to(src_root)
        out.mkdir(parents=True, exist_ok=True)
        frames = sorted(folder.glob("frame_*.jpg"))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda p: slim(p, out / p.name), frames))
        n_src += len(frames)
        n_dst += sum(1 for _ in out.glob("frame_*.jpg"))
    (dst_root / "windows.json").write_text(json.dumps(WINDOWS, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in dst_root.rglob("*.jpg")) / 1e9
    print(f"  {vid}: {n_dst}/{n_src} frames -> {dst_root} ({size:.2f} GB)")
    if n_dst != n_src:
        print(f"  {vid}: some frames were not copied - keeping the full frames")
        return False
    if delete_full:
        shutil.rmtree(src_root)
        print(f"  {vid}: deleted {src_root}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vods", nargs="+")
    parser.add_argument("--delete-full", action="store_true",
                        help="Delete ~/tft_frames_temp/VOD (incl. lobby/) once every game frame is copied")
    args = parser.parse_args()
    ok = all([slim_vod(v, args.delete_full) for v in args.vods])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
