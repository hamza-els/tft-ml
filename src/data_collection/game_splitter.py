"""
Detect game boundaries in a long TFT stream and split into individual game clips.

Strategy: HUD presence detection — sample one frame every N seconds and check
whether the gold AND HP regions contain active UI content. Transitions between
"in game" and "not in game" mark game start/end timestamps. ffmpeg then clips
the video at those timestamps without re-encoding (fast, lossless).

Asymmetric debouncing handles mid-game alt-tabs and screen switches:
  - Entering a game: 30s of active HUD required (catch game start quickly)
  - Exiting a game:  120s of inactive HUD required (don't split on brief alt-tabs)

Typical TFT stream: 5-6 hours, 8-10 games, each ~25-40 min.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class GameSegment:
    game_index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def duration_str(self) -> str:
        m, s = divmod(int(self.duration_sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def _region_active(frame: np.ndarray, region: tuple[int, int, int, int]) -> bool:
    """
    Returns True if a UI region looks like active game content.

    Active HUD elements are bright and high-contrast.
    Dark/uniform regions indicate a lobby, loading screen, or alt-tabbed desktop.
    """
    x, y, w, h = region
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    # Tune these thresholds after calibration on real frames
    return mean > 40 and std > 15


def _hud_active(
    frame: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
    require_all: bool = True,
) -> bool:
    """
    Check HUD presence across multiple regions.

    Args:
        require_all: If True, ALL regions must be active (used for game entry).
                     If False, ANY region active counts (used during game, handles pop-ups).
    """
    checks = [_region_active(frame, r) for r in regions.values()]
    return all(checks) if require_all else any(checks)


def find_game_segments(
    video_path: str,
    hud_regions: dict[str, tuple[int, int, int, int]],
    sample_interval_sec: float = 5.0,
    min_game_duration_sec: float = 900.0,
    enter_debounce_sec: float = 30.0,
    exit_debounce_sec: float = 120.0,
) -> list[GameSegment]:
    """
    Scan a VOD and return the timestamps of each detected TFT game.

    Uses asymmetric debouncing:
      - Entering game: requires enter_debounce_sec of active HUD (both regions)
      - Exiting game:  requires exit_debounce_sec of inactive HUD (both regions)
        This prevents brief alt-tabs or screen switches from splitting a game.

    Args:
        video_path: Path to the stream video file.
        hud_regions: Dict of named (x, y, w, h) crops to check for HUD presence.
                     Recommended: at least 'gold' and 'hp'.
        sample_interval_sec: Seconds between sampled frames. 5s is fast enough.
        min_game_duration_sec: Discard segments shorter than this.
        enter_debounce_sec: Consecutive seconds of active HUD to commit game start.
        exit_debounce_sec: Consecutive seconds of inactive HUD to commit game end.

    Returns:
        List of GameSegment objects, one per detected game.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    frame_step = int(fps * sample_interval_sec)

    enter_samples = max(1, int(enter_debounce_sec / sample_interval_sec))
    exit_samples = max(1, int(exit_debounce_sec / sample_interval_sec))

    total_samples = total_frames // frame_step
    print(f"Scanning {total_sec / 3600:.1f}h  |  "
          f"sample every {sample_interval_sec}s  |  "
          f"enter debounce {enter_debounce_sec}s  |  "
          f"exit debounce {exit_debounce_sec}s")

    in_game = False
    game_start_sec: float | None = None
    consecutive = 0
    segments: list[GameSegment] = []
    game_idx = 0
    frame_idx = 0

    bar = tqdm(total=total_samples, desc="Scanning", unit="sample", dynamic_ncols=True)

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        bar.set_postfix({"time": _fmt(current_sec), "state": "IN_GAME" if in_game else "LOBBY"})
        bar.update(1)

        if not in_game:
            # Waiting to enter a game: require ALL regions active
            if _hud_active(frame, hud_regions, require_all=True):
                consecutive += 1
                if consecutive >= enter_samples:
                    in_game = True
                    game_start_sec = current_sec - enter_debounce_sec
                    consecutive = 0
                    print(f"  >> Game {game_idx} START at {_fmt(game_start_sec)}")
            else:
                consecutive = 0
        else:
            # In a game: only exit if ALL regions are inactive (handles pop-ups
            # covering one region — stay in game as long as either is visible)
            if not _hud_active(frame, hud_regions, require_all=False):
                consecutive += 1
                if consecutive >= exit_samples:
                    end_sec = current_sec - exit_debounce_sec
                    duration = end_sec - (game_start_sec or 0)
                    in_game = False
                    consecutive = 0

                    if duration >= min_game_duration_sec:
                        segments.append(GameSegment(game_idx, game_start_sec or 0, end_sec))
                        print(f"  >> Game {game_idx} END at {_fmt(end_sec)} "
                              f"({duration / 60:.0f} min)")
                        game_idx += 1
                    else:
                        print(f"  >> Skipped short segment at {_fmt(game_start_sec or 0)} "
                              f"({duration / 60:.1f} min)")
                    game_start_sec = None
            else:
                consecutive = 0

        frame_idx += frame_step

    bar.close()

    # Game still running at stream end
    if in_game and game_start_sec is not None:
        duration = total_sec - game_start_sec
        if duration >= min_game_duration_sec:
            segments.append(GameSegment(game_idx, game_start_sec, total_sec))
            print(f"  >> Game {game_idx} ran to stream end ({duration / 60:.0f} min)")

    cap.release()
    print(f"\nFound {len(segments)} game(s).")
    return segments


def clip_games(
    video_path: str,
    segments: list[GameSegment],
    output_dir: str,
    padding_sec: float = 5.0,
) -> list[str]:
    """
    Use ffmpeg to cut each detected game from the stream.
    Stream-copies (no re-encoding) so it's very fast.

    Args:
        padding_sec: Extra seconds added before/after each segment as buffer.

    Returns:
        List of output file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    outputs = []

    for seg in segments:
        start = max(0.0, seg.start_sec - padding_sec)
        duration = seg.duration_sec + padding_sec * 2
        out_path = out_dir / f"{video_name}_game{seg.game_index:02d}.mp4"

        print(f"Clipping game {seg.game_index} ({seg.duration_str}) → {out_path.name}")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[-500:]}")
        else:
            outputs.append(str(out_path))

    return outputs


def _fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
