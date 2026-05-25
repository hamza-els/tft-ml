"""
Download Twitch VODs using yt-dlp.
Requires yt-dlp installed: pip install yt-dlp
"""

import subprocess
from pathlib import Path


def download_vod(url: str, output_dir: str, quality: str = "1080p") -> str:
    """
    Download a Twitch VOD to output_dir.

    Args:
        url: Twitch VOD URL (e.g. https://www.twitch.tv/videos/XXXXXXX)
        output_dir: Destination folder.
        quality: Target resolution string passed to yt-dlp format filter.

    Returns:
        Path to the downloaded file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(out_dir / "%(id)s.%(ext)s")
    format_str = f"bestvideo[height<={quality.replace('p', '')}]+bestaudio/best"

    cmd = [
        "yt-dlp",
        "--format", format_str,
        "--output", out_template,
        "--merge-output-format", "mp4",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # yt-dlp prints the final filename; parse it from stdout
    for line in result.stdout.splitlines():
        if "[Merger]" in line and "Merging formats into" in line:
            return line.split('"')[1]

    # Fallback: return the most recent mp4 in the output dir
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return str(mp4s[-1]) if mp4s else output_dir
