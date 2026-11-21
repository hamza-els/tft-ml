"""
Old first step vs the streaming one (src/data_collection/stream_extract.py) on
the same stretch of video: game boundaries, frames kept, time and disk.

Usage:
    python scripts/test_stream_extract.py VIDEO.mp4 --start 3600 --minutes 90

The stretch is cut out with stream copy (no re-encode) into the scratch folder,
then both run on that file: old = extract_frames -> find_game_segments ->
split_into_subfolders -> deduplicate_frames per game (as run_pipeline.py),
new = stream_extract.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import shutil
import subprocess
import time

import imageio_ffmpeg

from src.data_collection.frame_dedup import deduplicate_frames, frame_folders
from src.data_collection.frame_extractor import extract_frames
from src.data_collection.frame_game_splitter import find_game_segments, split_into_subfolders
from src.data_collection.stream_extract import stream_extract
from src.utils import paths
from src.vision.hud import HudDetector


def du(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*.jpg")) / 1e9


def kept(folder: Path) -> dict[str, set[int]]:
    return {g.name: {int(re.search(r"(\d+)", p.stem).group(1)) for p in g.glob("frame_*.jpg")}
            for g in sorted(folder.glob("game_*"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=3600)
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    root = paths.LOCAL / "stream_test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    clip = root / "clip.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(args.start), "-i", args.video,
                    "-t", str(args.minutes * 60), "-c", "copy", str(clip)], check=True)
    print(f"clip: {args.minutes:.0f} min from {args.start / 60:.0f} min -> {clip}")

    # old
    old = root / "old"
    t0 = time.time()
    extract_frames(str(clip), str(old), fps=2.0, quiet=True)
    t_extract = time.time() - t0
    peak_old = du(old)
    hud = HudDetector()
    segs = find_game_segments(str(old), hud, extraction_fps=2.0)
    if segs:
        split_into_subfolders(str(old), segs)
    t_split = time.time() - t0 - t_extract
    for folder in frame_folders(str(old)):
        deduplicate_frames(str(folder), quiet=True, hud=hud, max_gap_sec=5.0, fps=2.0,
                           combat_aware=True, watch_board=True)
    t_old = time.time() - t0
    print(f"OLD: {t_old / 60:.1f} min (extract {t_extract / 60:.1f}, split {t_split / 60:.1f}, "
          f"dedup {(t_old - t_extract - t_split) / 60:.1f}); disk peak {peak_old:.1f} GB, "
          f"after {du(old):.2f} GB (lobby kept in lobby/)")

    # new
    new = root / "new"
    t0 = time.time()
    summary = stream_extract(str(clip), str(new), workers=args.workers)
    t_new = time.time() - t0
    print(f"NEW: {t_new / 60:.1f} min (probe {summary['probe_s']:.0f} s); disk {du(new):.2f} GB, "
          f"decoded {summary['decoded_h']:.2f} of {summary['duration_h']:.2f} h")

    a, b = kept(old), kept(new)
    print(f"games: old {len(a)}, new {len(b)}")
    report = {"old_min": t_old / 60, "new_min": t_new / 60, "games": []}
    for name in sorted(set(a) | set(b)):
        sa, sb = a.get(name, set()), b.get(name, set())
        both = len(sa & sb)
        line = {"game": name, "old": len(sa), "new": len(sb), "same frame": both,
                "old range": [min(sa), max(sa)] if sa else None, "new range": [min(sb), max(sb)] if sb else None}
        # frames kept by one but not the other, but with the other keeping a frame within 1 s
        near = sum(1 for n in sa - sb if any(abs(n - m) <= 2 for m in sb))
        line["old-only but new kept one within 1 s"] = near
        report["games"].append(line)
        print(f"  {name}: old {len(sa)} kept, new {len(sb)}, same frame {both} "
              f"({both / max(len(sa), 1):.0%} of old); old-only with a new frame within 1 s: {near}; "
              f"ranges old {line['old range']} new {line['new range']}")
    (root / "comparison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
