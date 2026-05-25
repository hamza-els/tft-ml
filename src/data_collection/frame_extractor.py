"""
Extract frames from TFT VOD video files at a configurable interval.
Outputs frames to data/raw/frames/{video_name}/.
"""

import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list[str]:
    """
    Sample frames from a video at the given rate (frames per second of source video).

    Args:
        video_path: Path to the input .mp4 / .mkv file.
        output_dir: Directory to write extracted frame PNGs.
        fps: How many frames to extract per second of video. Default 1.0.

    Returns:
        List of file paths for the saved frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / fps))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = out_dir / f"frame_{saved_idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path))
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved
