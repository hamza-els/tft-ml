"""
Full data collection pipeline: download → extract → dedup → delete, one VOD at a time.
Reads URLs from data/raw/urls.txt. Only one VOD ever lives on disk at once.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --fps 2.0
    python scripts/run_pipeline.py --urls data/raw/urls.txt --fps 2.0 --no-dedup
"""

import sys
import ctypes
import subprocess
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import imageio_ffmpeg
from tqdm import tqdm

from src.data_collection.frame_extractor import extract_frames
from src.data_collection.frame_dedup import deduplicate_frames


URLS_FILE = Path("data/raw/urls.txt")
VOD_DIR   = Path("data/raw/vods")
FRAME_DIR = Path("data/raw/frames")
FORMAT    = "bestvideo[height<=1080][vcodec^=avc]+bestaudio/best"
W         = 60  # line width for dividers

ES_CONTINUOUS      = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _prevent_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def _allow_sleep():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _divider(char="─"):
    print(char * W)

def _step(label: str):
    print(f"\n  {label}")


def load_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def download(url: str, out_dir: Path, ffmpeg: str) -> Path | None:
    """Download a single VOD. Returns the downloaded file path or None on failure."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg,
        "--format", FORMAT,
        "--merge-output-format", "mp4",
        "--output", str(out_dir / "%(id)s.mp4"),
        "--no-overwrites",
        "--progress",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return None
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return mp4s[-1] if mp4s else None


def process_url(
    url: str,
    ffmpeg: str,
    fps: float,
    dedup: bool,
    idx: int,
    total: int,
    overall_bar: tqdm,
) -> bool:
    remaining_before = total - idx
    remaining_after  = total - idx  # same value, used in completion message

    _divider("═")
    print(f"  VOD {idx} of {total}  |  {remaining_before} remaining after this")
    print(f"  {url}")
    _divider("═")

    VOD_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Step 1 — Download
    overall_bar.set_description(f"VOD {idx}/{total} — downloading")
    _step("[ 1 / 3 ]  Downloading VOD")
    vod_path = download(url, VOD_DIR, ffmpeg)
    if vod_path is None:
        print("  Download failed — skipping.")
        return False

    size_gb = vod_path.stat().st_size / 1e9
    print(f"    {vod_path.name}  ({size_gb:.1f} GB)")

    # Step 2 — Extract frames
    overall_bar.set_description(f"VOD {idx}/{total} — extracting frames")
    _step(f"[ 2 / 3 ]  Extracting frames at {fps}fps")
    out_dir = FRAME_DIR / vod_path.stem
    saved = extract_frames(str(vod_path), str(out_dir), fps=fps)
    print(f"    {len(saved)} frames extracted")

    # Step 3 — Dedup
    if dedup and saved:
        overall_bar.set_description(f"VOD {idx}/{total} — deduplicating")
        _step("[ 3 / 3 ]  Deduplicating frames")
        kept, removed = deduplicate_frames(str(out_dir))
        print(f"    Kept {kept}  |  removed {removed}  "
              f"({removed / max(1, kept + removed) * 100:.0f}% dropped)")
    else:
        kept = len(saved)

    # Delete VOD
    vod_path.unlink()

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    _divider()
    print(f"  VOD {idx} done in {mins}m {secs}s  |  "
          f"{kept} frames kept  |  {remaining_after} VOD(s) remaining")

    overall_bar.update(1)
    overall_bar.set_description(f"VOD {idx}/{total} — complete")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=str(URLS_FILE))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    url_path = Path(args.urls)
    if not url_path.exists():
        print(f"URL file not found: {url_path}")
        return

    urls = load_urls(url_path)
    if not urls:
        print(f"No URLs in {url_path}. Add one URL per line.")
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dedup  = not args.no_dedup
    total  = len(urls)

    _divider("═")
    print(f"  TFT Pipeline  |  {total} VOD(s)  |  {args.fps}fps  |  dedup={dedup}")
    print(f"  Max local disk usage: ~1 VOD at a time (~3-5 GB)")
    _divider("═")

    _prevent_sleep()
    failed = []

    overall_bar = tqdm(
        total=total,
        desc=f"VOD 0/{total}",
        unit="VOD",
        position=0,
        dynamic_ncols=True,
        bar_format="{desc}  [{bar}]  {n}/{total}  {elapsed}<{remaining}",
    )

    try:
        for i, url in enumerate(urls, 1):
            try:
                ok = process_url(url, ffmpeg, args.fps, dedup, i, total, overall_bar)
                if not ok:
                    failed.append(url)
            except Exception as e:
                print(f"\n  Error on {url}: {e}")
                failed.append(url)
                overall_bar.update(1)
    finally:
        overall_bar.close()
        _allow_sleep()

    _divider("═")
    print(f"  All done.  {total - len(failed)}/{total} VODs processed successfully.")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for url in failed:
            print(f"    {url}")
    _divider("═")


if __name__ == "__main__":
    main()
