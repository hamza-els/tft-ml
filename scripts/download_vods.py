"""
Download TFT VODs listed in data/raw/urls.txt using yt-dlp.
Uses the imageio-ffmpeg bundled binary — no system ffmpeg install needed.

Usage:
    python scripts/download_vods.py
    python scripts/download_vods.py --urls data/raw/urls.txt
    python scripts/download_vods.py --urls data/raw/urls.txt --output-dir data/raw/vods
"""

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import imageio_ffmpeg


URLS_FILE = Path("data/raw/urls.txt")
OUTPUT_DIR = Path("data/raw/vods")
FORMAT = "bestvideo[height<=1080][vcodec^=avc]+bestaudio/best"


def load_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=str(URLS_FILE),
                        help=f"Path to URL list file (default: {URLS_FILE})")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help=f"Where to save downloaded VODs (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    url_path = Path(args.urls)
    if not url_path.exists():
        print(f"URL file not found: {url_path}")
        return

    urls = load_urls(url_path)
    if not urls:
        print(f"No URLs found in {url_path}. Add one URL per line and re-run.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"Found {len(urls)} URL(s) — downloading to {out_dir}/")
    print()

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg,
        "--format", FORMAT,
        "--merge-output-format", "mp4",
        "--output", str(out_dir / "%(id)s.mp4"),
        "--no-overwrites",          # skip if already downloaded
        "--progress",
        *urls,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nyt-dlp exited with errors — some downloads may have failed.")
    else:
        print(f"\nAll downloads complete. Run extract_frames.py next:")
        print(f"  python scripts/extract_frames.py --video-dir {out_dir} --fps 2.0 --dedup --delete-after --workers 2")


if __name__ == "__main__":
    main()
