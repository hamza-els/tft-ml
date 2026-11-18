"""
Process VODs end to end, one at a time: download, extract, split into games,
dedup, read states, collect icons, clean, encode, and cut unit crops.

Usage:
    python scripts/process_vods.py --urls data/raw/urls_batch1.txt

Each VOD goes through every step before the next one starts. A VOD whose
steps all finished is skipped on a re-run (data/processed/state_vectors/<id>/.done),
and run_pipeline.py itself resumes an interrupted download/extract/split/dedup.

Keeps the PC awake while it runs. Output for each step is appended to
data/raw/process_vods.log.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import ctypes
import os
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FRAMES = Path.home() / "tft_frames_temp"
STATES = ROOT / "data" / "processed" / "state_vectors"
LOG = ROOT / "data" / "raw" / "process_vods.log"

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


class BotCheck(Exception):
    """YouTube asked to confirm we're not a bot: every later download will fail too."""


def _log_tail(chars: int = 20000) -> str:
    with open(LOG, encoding="utf-8", errors="replace") as f:
        f.seek(max(0, f.seek(0, 2) - chars))
        return f.read()


def process(url: str) -> bool:
    vid = video_id(url)
    frames, states = FRAMES / vid, STATES / vid
    if (states / ".done").exists():
        print(f"  {vid}: already done, skipping", flush=True)
        return True
    print(f"  {vid}", flush=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(url + "\n")
        single = f.name
    try:
        if not run("download + split + dedup", ["scripts/run_pipeline.py", "--urls", single, *COOKIES]):
            if "not a bot" in _log_tail():
                raise BotCheck()
            return False
    finally:
        os.unlink(single)
    if not any(frames.glob("game_*")):
        print(f"    no games found in {vid} - skipping the rest", flush=True)
        return False

    steps = [
        ("read states", ["scripts/extract_states.py", "--frames", str(frames)]),
        ("augment icons", ["scripts/harvest_augment_icons.py", "--frames", str(frames), "--update-states", str(states)]),
        ("item icons", ["scripts/harvest_item_icons.py", "--frames", str(frames), "--update-states", str(states)]),
        ("clean states", ["scripts/clean_states.py", "--states", str(states)]),
        ("encode dataset", ["scripts/build_dataset.py", "--states", str(states)]),
        ("unit crops", ["scripts/collect_unit_crops.py", "--frames", str(frames), "--states", str(states)]),
    ]
    for step, args in steps:
        if not run(step, args):
            return False
    (states / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", required=True)
    parser.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                        help="Passed to run_pipeline.py (e.g. firefox)")
    args = parser.parse_args()
    if args.cookies_from_browser:
        COOKIES.extend(["--cookies-from-browser", args.cookies_from_browser])

    urls = [l.strip() for l in Path(args.urls).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        failed = []
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {time.strftime('%H:%M')}", flush=True)
            try:
                ok = process(url)
            except BotCheck:
                print("    YouTube is asking to confirm we're not a bot. Stopping here; wait a few hours\n"
                      "    and re-run the same command (finished VODs are skipped), or add\n"
                      "    --cookies-from-browser firefox.", flush=True)
                failed.extend(urls[i - 1:])
                break
            if not ok:
                failed.append(url)
        print(f"\nDone. {len(urls) - len(failed)}/{len(urls)} VODs processed.", flush=True)
        for url in failed:
            print(f"  failed: {url}  (details in {LOG})", flush=True)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
