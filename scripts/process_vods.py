"""
Process VODs end to end, one at a time: download, extract, split into games,
dedup, read states, collect icons, read board/bench units, clean, encode,
and cut unit crops.

Usage:
    python scripts/process_vods.py --urls data/raw/urls_batch1.txt

Each VOD goes through every step before the next one starts. A VOD whose
steps all finished is skipped on a re-run (~/tft_data/state_vectors/<id>/.done),
and run_pipeline.py itself resumes an interrupted download/extract/split/dedup.

Everything happens on the local disk (src/utils/paths.py); the repo on
Google Drive only receives the finished state files, in the last step
(scripts/upload_states.py). Exceptions: the icon harvest steps add template
PNGs to assets/templates/, and unit crops add entries to the repo's
data/processed/unit_crops/index.json (the images stay local).

Keeps the PC awake while it runs. Output for each step is appended to
~/tft_data/logs/process_vods.log.

Options for long batches:
    --slim-frames         after a VOD is done, keep only augment-only copies of its
                          frames (scripts/slim_frames.py, ~0.3 GB instead of ~6 GB) and
                          delete the full frames; augments can still be redone later
                          (scripts/redo_augments.py), units/OCR need a re-download.
                          With --keep-frames-gb N, full frames are kept instead while N GB
                          would still be free.
    --skip-augment-icons  don't harvest augment templates / rewrite augments in this
                          batch (do it later with scripts/redo_augments.py).
    --wait-on-bot-check H when YouTube asks to confirm we're not a bot, wait H hours
                          and carry on instead of stopping the batch.
    --no-augment-rebuild  at the end of the batch, don't rebuild the augment templates from the
                          newly named games (scripts/apply_augment_names.py: templates from every
                          game named by label or overlay, then every finished VOD read again).
    --no-prefetch         don't download the next VOD in the background. By default,
                          once a VOD's frames are split and deduped, the next VOD's video
                          file is downloaded while this one is read (log:
                          ~/tft_data/logs/prefetch.log), if --min-free-gb + --prefetch-gb
                          (default 30 + 25 GB) is free; at most two VODs are on disk.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import ctypes
import os
import shutil
import subprocess
import tempfile
import time

from src.utils import paths

ROOT = Path(__file__).resolve().parents[1]
FRAMES = paths.FRAMES
STATES = paths.STATES                      # local working copy
LOG = paths.LOGS / "process_vods.log"

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def video_id(url: str) -> str:
    return url.rstrip("/").split("=")[-1].split("/")[-1]


def run(step: str, args: list[str]) -> bool:
    start = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {step}\n")
        log.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    mins = (time.time() - start) / 60
    status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"    {step:<22} {status}  ({mins:.0f} min)", flush=True)
    return result.returncode == 0


COOKIES: list[str] = []   # extra run_pipeline arguments, set from --cookies-from-browser
JOBS = [3]                # games read in parallel by the state and unit readers (--jobs)
OPTS = {"stream": True, "stream_read": True, "augment_rebuild": True, "skip_augments": False, "slim": False, "keep_frames_gb": 0.0, "min_free_gb": 0.0, "prefetch": True, "prefetch_gb": 25.0}
PREFETCH_LOG = paths.LOGS / "prefetch.log"
_prefetch: dict = {}      # the next VOD's background download: {"url", "proc", "log", "start"}


def start_prefetch(url: str | None):
    """
    Download `url` in the background (run_pipeline.py --download-only) while
    the current VOD is read, so its download time is hidden. Only the video
    file is fetched (~7 GB), not extracted; extraction, split and dedup run
    in the VOD's own turn. Skipped when the VOD is done or its frames exist,
    or when disk space is short.
    """
    if not url or not OPTS["prefetch"] or _prefetch:
        return
    vid = video_id(url)
    if (STATES / vid / ".done").exists() or (FRAMES / vid).exists():
        return
    # Disk is the batch's limit: the download needs room for itself (a 9 h VOD's mp4 is
    # ~11.5 GB; video only since 2026-11-24, so no merge copy any more) on top of what the next VOD's
    # own turn still wants free (--min-free-gb).
    free_gb = shutil.disk_usage(FRAMES.parent).free / 1e9
    if free_gb < OPTS["min_free_gb"] + OPTS["prefetch_gb"]:
        print(f"    not downloading {vid} in the background: {free_gb:.0f} GB free, "
              f"{OPTS['min_free_gb'] + OPTS['prefetch_gb']:.0f} GB wanted", flush=True)
        return
    log = open(PREFETCH_LOG, "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  download {vid}\n")
    log.flush()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(url + "\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "scripts/run_pipeline.py", "--urls", f.name, "--download-only", *COOKIES],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    _prefetch.update(url=url, proc=proc, log=log, start=time.time(), urls_file=f.name)
    print(f"    downloading {vid} in the background", flush=True)


def finish_prefetch(url: str | None = None, stop: bool = False):
    """Wait for the background download (of `url`, or any); stop=True cancels it instead."""
    if not _prefetch or (url and _prefetch["url"] != url):
        return
    proc = _prefetch["proc"]
    if stop:
        proc.terminate()
    elif proc.poll() is None:
        print(f"    waiting for the background download of {video_id(_prefetch['url'])}", flush=True)
    proc.wait()
    _prefetch["log"].close()
    os.unlink(_prefetch["urls_file"])
    if not stop:
        mins = (time.time() - _prefetch["start"]) / 60
        status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode}); retried in its own turn"
        print(f"    background download {status}  ({mins:.0f} min after it started)", flush=True)
    _prefetch.clear()


class LowDisk(Exception):
    """Not enough free disk space to download and extract another VOD."""


class BotCheck(Exception):
    """YouTube asked to confirm we're not a bot: every later download will fail too."""


def _log_tail(chars: int = 20000) -> str:
    with open(LOG, encoding="utf-8", errors="replace") as f:
        f.seek(max(0, f.seek(0, 2) - chars))
        return f.read()


def process(url: str, next_url: str | None = None) -> bool:
    vid = video_id(url)
    frames, states = FRAMES / vid, STATES / vid
    if (states / ".done").exists():
        print(f"  {vid}: already done, skipping", flush=True)
        return True
    print(f"  {vid}", flush=True)
    finish_prefetch(url)      # its download may have started during the previous VOD

    free_gb = shutil.disk_usage(FRAMES.parent).free / 1e9
    # A video file already downloaded in the background is part of what min_free_gb is for.
    mp4 = Path.home() / "tft_vods_temp" / f"{vid}.mp4"
    have_gb = mp4.stat().st_size / 1e9 if mp4.exists() else 0.0
    if not frames.exists() and free_gb + have_gb < OPTS["min_free_gb"]:
        raise LowDisk(f"{free_gb:.0f} GB free, {OPTS['min_free_gb'] - have_gb:.0f} GB wanted before a download")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(url + "\n")
        single = f.name
    try:
        first = ["--stream"] if OPTS["stream"] else []
        if OPTS["stream"] and OPTS["stream_read"]:
            # read each game in the same pass, while the rest of the video decodes
            first += ["--stream-read", str(JOBS[0])]
        if not run("download + split + dedup", ["scripts/run_pipeline.py", "--urls", single, *first, *COOKIES]):
            if "not a bot" in _log_tail():
                raise BotCheck()
            return False
    finally:
        os.unlink(single)
    if not any(frames.glob("game_*")):
        print(f"    no games found in {vid} - skipping the rest", flush=True)
        return False
    # This VOD's own download is over: fetch the next one while this one is read.
    start_prefetch(next_url)

    steps = [
        ("read states", ["scripts/extract_states.py", "--frames", str(frames), "--jobs", str(JOBS[0])]),
        # augment names from the stream overlay; needs the full frames, so before any slimming
        ("augment overlay", ["scripts/read_augment_overlay.py", "--", vid]),
        ("item icons", ["scripts/harvest_item_icons.py", "--frames", str(frames), "--update-states", str(states)]),
        ("read units", ["scripts/read_units.py", "--frames", str(frames), "--states", str(states), "--jobs", str(JOBS[0])]),
        ("clean states", ["scripts/clean_states.py", "--states", str(states)]),
        ("encode dataset", ["scripts/build_dataset.py", "--states", str(states)]),
        ("unit crops", ["scripts/collect_unit_crops.py", "--frames", str(frames), "--states", str(states)]),
        ("upload states", ["scripts/upload_states.py", "--", vid]),
    ]
    if OPTS["skip_augments"]:
        steps = [s for s in steps if s[0] != "augment icons"]
    if (states / ".stream_read").exists():
        # every game was already read in the streaming pass (src/data_collection/stream_extract.py)
        print("    states, augment overlay and units read in the streaming pass", flush=True)
        steps = [s for s in steps if s[0] not in ("read states", "augment overlay", "read units")]
    for step, args in steps:
        if not run(step, args):
            return False
    (states / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    if OPTS["slim"]:
        free_gb = shutil.disk_usage(FRAMES.parent).free / 1e9
        if free_gb >= OPTS["keep_frames_gb"]:
            print(f"    keeping full frames ({free_gb:.0f} GB free, slimming below {OPTS['keep_frames_gb']:.0f})",
                  flush=True)
        else:
            run("slim frames", ["scripts/slim_frames.py", "--delete-full", "--", vid])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True)
    parser.add_argument("--jobs", type=int, default=3,
                        help="Games read in parallel by the state and unit readers (default 3)")
    parser.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                        help="Passed to run_pipeline.py (e.g. firefox)")
    parser.add_argument("--slim-frames", action="store_true",
                        help="After each VOD: keep augment-only frame copies, delete the full frames")
    parser.add_argument("--keep-frames-gb", type=float, default=0, metavar="GB",
                        help="With --slim-frames: keep a VOD's full frames instead while at least this much "
                             "disk is still free after it (e.g. 45: room for a 9 h VOD's ~34 GB peak)")
    parser.add_argument("--skip-augment-icons", action="store_true",
                        help="Skip the augment harvest step (redo later with scripts/redo_augments.py)")
    parser.add_argument("--wait-on-bot-check", type=float, default=0, metavar="HOURS",
                        help="On YouTube's bot check, wait this long and retry instead of stopping")
    parser.add_argument("--min-free-gb", type=float, default=30,
                        help="Stop before downloading when less disk is free (default 30; a 9 h VOD peaks at ~34 GB)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Old first step (extract every frame, split, dedup) instead of the streaming pass")
    parser.add_argument("--no-stream-read", action="store_true",
                        help="Stream only finds games and keeps frames; the step scripts read them after")
    parser.add_argument("--no-augment-rebuild", action="store_true",
                        help="Don't rebuild the augment templates and re-read every VOD at the end of the batch")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Don't download the next VOD in the background while the current one is read")
    parser.add_argument("--prefetch-gb", type=float, default=25,
                        help="Download the next VOD in the background only when this much more than "
                             "--min-free-gb is free (default 25: set when yt-dlp's audio merge briefly held two "
                             "copies of the mp4; video-only downloads need about half)")
    args = parser.parse_args()
    JOBS[0] = args.jobs
    OPTS.update(skip_augments=args.skip_augment_icons, slim=args.slim_frames, keep_frames_gb=args.keep_frames_gb,
                min_free_gb=args.min_free_gb,
                prefetch=not args.no_prefetch, augment_rebuild=not args.no_augment_rebuild, stream=not args.no_stream, stream_read=not args.no_stream_read, prefetch_gb=args.prefetch_gb)
    if args.cookies_from_browser:
        COOKIES.extend(["--cookies-from-browser", args.cookies_from_browser])

    urls = [l.strip() for l in Path(args.urls).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        failed = []
        i, waits = 0, 0
        while i < len(urls):
            url = urls[i]
            i += 1
            print(f"[{i}/{len(urls)}] {time.strftime('%H:%M')}", flush=True)
            try:
                ok = process(url, urls[i] if i < len(urls) else None)
            except BotCheck:
                if args.wait_on_bot_check and waits < 10:
                    waits += 1
                    print(f"    YouTube is asking to confirm we're not a bot. Waiting "
                          f"{args.wait_on_bot_check:g} h, then retrying (wait {waits}/10).", flush=True)
                    time.sleep(args.wait_on_bot_check * 3600)
                    i -= 1
                    continue
                print("    YouTube is asking to confirm we're not a bot. Stopping here; wait a few hours\n"
                      "    and re-run the same command (finished VODs are skipped), or add\n"
                      "    --cookies-from-browser firefox.", flush=True)
                failed.extend(urls[i - 1:])
                break
            except LowDisk as e:
                print(f"    Stopping: {e}. Free some space and re-run the same command.", flush=True)
                failed.extend(urls[i - 1:])
                break
            if not ok:
                failed.append(url)
        print(f"\nDone. {len(urls) - len(failed)}/{len(urls)} VODs processed.", flush=True)
        for url in failed:
            print(f"  failed: {url}  (details in {LOG})", flush=True)
        if OPTS["augment_rebuild"] and len(failed) < len(urls):
            # Grow the augment templates from the games the overlay or labels just named, and
            # read every finished VOD again with them: streams without the overlay depend on them.
            run("augment templates", ["scripts/apply_augment_names.py"])
    finally:
        finish_prefetch(stop=True)    # batch stopped early: don't leave a download running
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
