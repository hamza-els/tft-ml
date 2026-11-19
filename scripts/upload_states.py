"""
Copy finished state vectors from the local working folder into the repo
(the Google Drive folder). This is the only step that writes VOD data there.

Usage:
    python scripts/upload_states.py VOD_ID [VOD_ID ...]
    python scripts/upload_states.py --all

Copies ~/tft_data/state_vectors/<vod>/game_XX.jsonl and game_XX.clean.jsonl
to data/processed/state_vectors/<vod>/. Files are written under a temporary
name and renamed, retrying while Drive's sync holds a lock on them.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
import time

from src.utils import paths


def copy_with_retry(src: Path, dst: Path, tries: int = 30, wait: float = 2.0):
    tmp = dst.with_name(dst.name + ".part")
    for i in range(tries):
        try:
            shutil.copyfile(src, tmp)
            tmp.replace(dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def upload(vod: str) -> int:
    src_dir, dst_dir = paths.STATES / vod, paths.REPO_STATES / vod
    files = sorted(src_dir.glob("game_*.jsonl"))
    if not files:
        print(f"  {vod}: no state files in {src_dir}")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        copy_with_retry(f, dst_dir / f.name)
    print(f"  {vod}: {len(files)} files -> {dst_dir}")
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vods", nargs="*")
    parser.add_argument("--all", action="store_true", help="Every VOD in the local working folder")
    args = parser.parse_args()
    vods = sorted(p.name for p in paths.STATES.iterdir() if p.is_dir()) if args.all else args.vods
    if not vods:
        sys.exit("Name VOD ids or pass --all")
    for vod in vods:
        upload(vod)


if __name__ == "__main__":
    main()
