"""
Full data collection pipeline: download → extract → split into games → dedup,
one VOD at a time. Reads URLs from data/raw/urls.txt. Frames stay on local
disk in LOCAL_TEMP/<video_id>/game_XX/ (non-game frames go to lobby/).

Each finished step is recorded in LOCAL_TEMP/<video_id>/.pipeline_steps, so
re-running after an interruption resumes at the first unfinished step.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --fps 2.0 --workers 2 --fragments 4
    python scripts/run_pipeline.py --no-dedup
    python scripts/run_pipeline.py --no-split     # keep one flat folder per VOD
"""

import sys
import ctypes
import subprocess
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
from tqdm import tqdm

_bar_lock = threading.Lock()

from src.data_collection.frame_extractor import extract_frames
from src.data_collection.frame_dedup import deduplicate_frames, frame_folders
from src.data_collection.frame_game_splitter import find_game_segments, split_into_subfolders
from src.vision.hud import HudDetector


URLS_FILE  = Path("data/raw/urls.txt")
VOD_DIR    = Path.home() / "tft_vods_temp"    # local — never inside Google Drive folder
LOCAL_TEMP = Path.home() / "tft_frames_temp"
STEPS_FILE = ".pipeline_steps"   # one finished step name per line, inside each VOD folder
FORMAT     = "bestvideo[height<=1080][vcodec^=avc]+bestaudio/best"
W          = 60

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


def download(url: str, out_dir: Path, ffmpeg: str, fragments: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # yt-dlp always writes %(id)s.mp4, so we know the path before downloading.
    # Using set-difference was racy when two workers download simultaneously.
    video_id = url.rstrip("/").split("=")[-1].split("/")[-1]
    out_path = out_dir / f"{video_id}.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg,
        # YouTube needs a JS runtime to solve its challenge; without one
        # yt-dlp falls back to a client whose URLs 403 mid-download.
        "--js-runtimes", "node",
        "--format", FORMAT,
        "--merge-output-format", "mp4",
        "--output", str(out_dir / "%(id)s.mp4"),
        "--no-overwrites",
        "--concurrent-fragments", str(fragments),
        "--progress",
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return None
    return out_path if out_path.exists() else None


def _bar_update(bar: tqdm, desc: str | None = None, increment: int = 0):
    """Thread-safe tqdm update."""
    with _bar_lock:
        if desc:
            bar.set_description(desc)
        if increment:
            bar.update(increment)
        bar.refresh()


def _steps_done(local_dir: Path) -> set[str]:
    marker = local_dir / STEPS_FILE
    return set(marker.read_text().split()) if marker.exists() else set()


def _mark_done(local_dir: Path, step: str):
    with open(local_dir / STEPS_FILE, "a") as f:
        f.write(step + "\n")


def process_url(
    url: str,
    ffmpeg: str,
    fps: float,
    dedup: bool,
    split: bool,
    idx: int,
    total: int,
    overall_bar: tqdm,
    quiet: bool = False,
    fragments: int = 4,
) -> bool:
    def log(msg: str):
        if not quiet:
            print(msg)

    if not quiet:
        _divider("═")
        print(f"  VOD {idx} of {total}  |  {total - idx} remaining after this")
        print(f"  {url}")
        _divider("═")

    # ── Already done? ─────────────────────────────────────────────
    video_id = url.rstrip("/").split("=")[-1].split("/")[-1]
    local_dir = LOCAL_TEMP / video_id
    wanted = ["extract"] + (["split"] if split else []) + (["dedup"] if dedup else [])
    done = _steps_done(local_dir)
    # Folders from before step tracking existed: frames on disk means extraction finished.
    if "extract" not in done and local_dir.exists() and any(local_dir.rglob("frame_*.jpg")):
        done.add("extract")
        _mark_done(local_dir, "extract")
    if all(step in done for step in wanted):
        n = sum(1 for _ in local_dir.rglob("frame_*.jpg"))
        tqdm.write(f"  ↩ VOD {idx}/{total} — {video_id} — already processed ({n:,} frames), skipping")
        _bar_update(overall_bar, desc=f"VOD {idx}/{total} — skipped (done)", increment=1)
        return True

    t_start = time.time()

    if "extract" not in done:
        # ── Step 1: Download ──────────────────────────────────────
        _bar_update(overall_bar, desc=f"VOD {idx}/{total} — downloading")
        log(f"\n  [ 1 / 4 ]  Downloading VOD")
        vod_path = download(url, VOD_DIR, ffmpeg, fragments)
        if vod_path is None:
            tqdm.write(f"  VOD {idx} — download failed, skipping.")
            _bar_update(overall_bar, increment=1)
            return False
        log(f"    {vod_path.name}  ({vod_path.stat().st_size / 1e9:.1f} GB)")

        # ── Step 2: Extract to LOCAL disk ─────────────────────────
        _bar_update(overall_bar, desc=f"VOD {idx}/{total} — extracting")
        log(f"\n  [ 2 / 4 ]  Extracting frames → local disk")

        def _extract_progress(n: int):
            _bar_update(overall_bar, desc=f"VOD {idx}/{total} — extracting ({n:,} frames)")

        saved = extract_frames(str(vod_path), str(local_dir), fps=fps, quiet=quiet,
                               on_progress=_extract_progress if quiet else None)
        log(f"    {len(saved)} frames extracted")
        if not saved:
            tqdm.write(f"  VOD {idx} — no frames extracted, skipping.")
            _bar_update(overall_bar, increment=1)
            return False
        vod_path.unlink()
        _mark_done(local_dir, "extract")

    hud = HudDetector()

    # ── Step 3: Split into games ──────────────────────────────────
    # Before dedup: boundaries are found from the HUD, and lobby frames
    # are moved out so dedup doesn't waste time on them.
    if split and "split" not in done:
        _bar_update(overall_bar, desc=f"VOD {idx}/{total} — splitting games")
        log(f"\n  [ 3 / 4 ]  Splitting into games")
        if any(local_dir.glob("frame_*.jpg")):
            segments = find_game_segments(str(local_dir), hud, extraction_fps=fps)
            if segments:
                split_into_subfolders(str(local_dir), segments)
            else:
                log("    No games detected — leaving frames unsplit "
                    "(check the hud section of configs/vision_config.yaml)")
        _mark_done(local_dir, "split")

    # ── Step 4: Dedup locally ─────────────────────────────────────
    if dedup and "dedup" not in done:
        _bar_update(overall_bar, desc=f"VOD {idx}/{total} — deduping")
        log(f"\n  [ 4 / 4 ]  Deduplicating frames")
        # Safe to restart after an interruption: it makes the same choices again.
        for folder in frame_folders(str(local_dir)):
            log(f"    {folder.name}")
            deduplicate_frames(str(folder), quiet=quiet, hud=hud, max_gap_sec=5.0, fps=fps)
        _mark_done(local_dir, "dedup")

    kept = sum(len(list(f.glob("frame_*.jpg"))) for f in frame_folders(str(local_dir)))
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    tqdm.write(f"  ✓ VOD {idx}/{total} — {video_id} — "
               f"{kept} frames → {local_dir}  ({mins}m {secs}s)")

    _bar_update(overall_bar, desc=f"VOD {idx}/{total} complete", increment=1)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=str(URLS_FILE))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-split", action="store_true",
                        help="Don't split frames into per-game folders")
    parser.add_argument("--workers", type=int, default=1,
                        help="VODs to process in parallel (default: 1, max 2 recommended)")
    parser.add_argument("--fragments", type=int, default=4,
                        help="Concurrent download fragments per VOD (default: 4)")
    args = parser.parse_args()

    url_path = Path(args.urls)
    if not url_path.exists():
        print(f"URL file not found: {url_path}")
        return

    urls = load_urls(url_path)
    if not urls:
        print(f"No URLs in {url_path}.")
        return

    ffmpeg  = imageio_ffmpeg.get_ffmpeg_exe()
    dedup   = not args.no_dedup
    split   = not args.no_split
    total   = len(urls)
    workers = args.workers
    quiet   = workers > 1

    _divider("═")
    print(f"  TFT Pipeline  |  {total} VOD(s)  |  {args.fps}fps  |  "
          f"workers={workers}  |  split={split}  |  dedup={dedup}")
    print(f"  VODs (temp):  {VOD_DIR}")
    print(f"  Frames:       {LOCAL_TEMP}")
    _divider("═")

    VOD_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
    _prevent_sleep()
    failed = []

    overall_bar = tqdm(
        total=total,
        desc=f"VOD 0/{total}",
        unit="VOD",
        position=0,
        dynamic_ncols=True,
        bar_format="{desc}  [{bar}]  {n}/{total}  {elapsed}<{remaining}",
        smoothing=0,
        mininterval=1.0,
    )

    try:
        if workers == 1:
            for i, url in enumerate(urls, 1):
                try:
                    ok = process_url(url, ffmpeg, args.fps, dedup, split, i, total,
                                     overall_bar, quiet=False, fragments=args.fragments)
                    if not ok:
                        failed.append(url)
                except Exception as e:
                    print(f"\n  Error: {e}")
                    failed.append(url)
                    overall_bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_url, url, ffmpeg, args.fps, dedup, split,
                                i, total, overall_bar, True, args.fragments): url
                    for i, url in enumerate(urls, 1)
                }
                for future in as_completed(futures):
                    url = futures[future]
                    try:
                        if not future.result():
                            failed.append(url)
                    except Exception as e:
                        print(f"\n  Error: {e}")
                        failed.append(url)
    finally:
        overall_bar.close()
        _allow_sleep()

    _divider("═")
    print(f"  All done.  {total - len(failed)}/{total} VODs processed.")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for url in failed:
            print(f"    {url}")
    _divider("═")


if __name__ == "__main__":
    main()
