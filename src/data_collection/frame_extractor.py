"""
Extract frames from TFT VOD video files at a configurable interval.
Uses ffmpeg via imageio-ffmpeg (bundled binary, no PATH setup needed).
"""

import subprocess
import threading
from pathlib import Path

import imageio_ffmpeg
from tqdm import tqdm


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_duration_sec(video_path: str) -> float | None:
    """Return video duration in seconds by parsing ffmpeg's stderr output."""
    result = subprocess.run(
        [_ffmpeg(), "-i", video_path, "-hide_banner"],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            parts = line.strip().split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0, quiet: bool = False, on_progress=None) -> list[str]:
    """
    Sample frames from a video at the given rate using ffmpeg.

    Args:
        video_path: Path to the input .mp4 / .mkv file.
        output_dir: Directory to write extracted frames (saved as JPEG).
        fps: How many frames to extract per second of video. Default 1.0.

    Returns:
        List of file paths for the saved frames.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%06d.jpg")

    duration = _get_duration_sec(video_path)
    expected_frames = int((duration or 0) * fps)

    cmd = [
        _ffmpeg(), "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-hide_banner",
        "-loglevel", "warning",
        "-progress", "pipe:1",  # write progress key=value to stdout
        "-nostats",
        out_pattern,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Drain stderr in a background thread to prevent the OS pipe buffer (~64 KB)
    # from filling up and deadlocking ffmpeg before it can write more stdout progress.
    stderr_lines: list[str] = []
    def _drain_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
    threading.Thread(target=_drain_stderr, daemon=True).start()

    bar = tqdm(
        total=expected_frames,
        desc=f"  Extracting frames",
        unit="frame",
        dynamic_ncols=True,
        disable=quiet,
        smoothing=0,       # ETA from cumulative average, not recent window
        mininterval=1.0,   # redraw at most once per second
    )

    last_frame = 0
    for line in process.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=")[1])
                bar.update(current_frame - last_frame)
                last_frame = current_frame
                if on_progress and current_frame % 200 == 0:
                    on_progress(current_frame)
            except ValueError:
                pass

    process.wait()
    bar.close()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{''.join(stderr_lines)}")

    saved = sorted(str(p) for p in out_dir.glob("frame_*.jpg"))
    return saved


def extract_single_frame(video_path: str, timestamp_sec: float, output_path: str) -> str:
    """
    Extract one frame at a specific timestamp. Useful for grabbing template images.

    Args:
        timestamp_sec: Time in seconds (e.g. 125.0 = 2 min 5 sec).
        output_path: Full path for the output file (e.g. 'assets/templates/ui/level/6.jpg').

    Returns:
        Output file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg(), "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-hide_banner",
        "-loglevel", "warning",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return output_path
