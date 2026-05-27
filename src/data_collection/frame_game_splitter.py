"""
Split already-extracted frames into per-game subfolders using HUD detection.
Works on frames that were extracted from multi-game streams.

Frames are moved (not copied) into subfolders:
  data/raw/frames/VOD_ID/game_00/
  data/raw/frames/VOD_ID/game_01/
  ...

Frames outside detected games (lobby, loading screens) are moved to:
  data/raw/frames/VOD_ID/lobby/

Uses the same asymmetric debouncing logic as game_splitter.py but
operates on frame files instead of seeking through a video.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class FrameGameSegment:
    game_index: int
    frame_paths: list[Path]

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


def _region_active(frame: np.ndarray, region: tuple[int, int, int, int]) -> bool:
    x, y, w, h = region
    crop = frame[y:y + h, x:x + w].astype(float)
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
    return float(gray.mean()) > 40 and float(gray.std()) > 15


def _hud_active(
    frame: np.ndarray,
    regions: dict[str, tuple[int, int, int, int]],
    require_all: bool = True,
) -> bool:
    checks = [_region_active(frame, r) for r in regions.values()]
    return all(checks) if require_all else any(checks)


def find_game_segments(
    frame_dir: str,
    hud_regions: dict[str, tuple[int, int, int, int]],
    extraction_fps: float = 2.0,
    min_game_duration_sec: float = 900.0,
    enter_debounce_sec: float = 30.0,
    exit_debounce_sec: float = 120.0,
) -> list[FrameGameSegment]:
    """
    Scan extracted frames and group them into game segments.

    Args:
        frame_dir: Directory containing frame_XXXXXX.jpg files.
        hud_regions: Dict of (x, y, w, h) regions to check for HUD presence.
        extraction_fps: FPS used when frames were extracted (default 2.0).
        min_game_duration_sec: Discard segments shorter than this.
        enter_debounce_sec: Seconds of active HUD to commit game start.
        exit_debounce_sec: Seconds of inactive HUD to commit game end.

    Returns:
        List of FrameGameSegment objects, one per detected game.
    """
    paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    min_game_frames  = int(min_game_duration_sec * extraction_fps)
    enter_debounce   = max(1, int(enter_debounce_sec * extraction_fps))
    exit_debounce    = max(1, int(exit_debounce_sec * extraction_fps))
    total_sec        = len(paths) / extraction_fps

    print(f"Scanning {len(paths)} frames  ({total_sec / 3600:.1f}h at {extraction_fps}fps)  |  "
          f"enter debounce {enter_debounce_sec}s  |  exit debounce {exit_debounce_sec}s")

    in_game         = False
    game_start_idx  = None
    consecutive     = 0
    segments        = []
    game_idx        = 0
    current_game_frames: list[Path] = []

    for i, path in enumerate(tqdm(paths, desc="Scanning frames", unit="frame", dynamic_ncols=True,
                                  smoothing=0, mininterval=1.0)):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        if not in_game:
            if _hud_active(frame, hud_regions, require_all=True):
                consecutive += 1
                if consecutive >= enter_debounce:
                    in_game = True
                    game_start_idx = i - enter_debounce
                    current_game_frames = list(paths[max(0, game_start_idx):i + 1])
                    consecutive = 0
                    print(f"\n  >> Game {game_idx} START at frame {game_start_idx} "
                          f"({game_start_idx / extraction_fps / 60:.1f} min)")
            else:
                consecutive = 0
        else:
            current_game_frames.append(path)
            if not _hud_active(frame, hud_regions, require_all=False):
                consecutive += 1
                if consecutive >= exit_debounce:
                    end_idx = i - exit_debounce
                    game_frames = current_game_frames[:end_idx - game_start_idx + 1]

                    if len(game_frames) >= min_game_frames:
                        segments.append(FrameGameSegment(game_idx, game_frames))
                        print(f"\n  >> Game {game_idx} END at frame {end_idx}  "
                              f"({len(game_frames)} frames  |  "
                              f"{len(game_frames) / extraction_fps / 60:.0f} min)")
                        game_idx += 1
                    else:
                        print(f"\n  >> Skipped short segment "
                              f"({len(game_frames) / extraction_fps / 60:.1f} min)")

                    in_game = False
                    game_start_idx = None
                    current_game_frames = []
                    consecutive = 0
            else:
                consecutive = 0

    # Game still running at end of frames
    if in_game and current_game_frames and len(current_game_frames) >= min_game_frames:
        segments.append(FrameGameSegment(game_idx, current_game_frames))
        print(f"\n  >> Game {game_idx} ran to end of frames "
              f"({len(current_game_frames)} frames)")

    print(f"\nFound {len(segments)} game(s).")
    return segments


def split_into_subfolders(
    frame_dir: str,
    segments: list[FrameGameSegment],
    move_lobby: bool = True,
) -> list[str]:
    """
    Move frames into per-game subfolders inside frame_dir.

    Args:
        frame_dir: Parent directory containing all frames.
        segments: Output of find_game_segments().
        move_lobby: If True, move non-game frames into a lobby/ subfolder.
                    If False, delete them.

    Returns:
        List of created game subfolder paths.
    """
    base = Path(frame_dir)
    game_frame_set: set[Path] = set()
    created = []

    for seg in segments:
        dest = base / f"game_{seg.game_index:02d}"
        dest.mkdir(exist_ok=True)
        print(f"Moving {seg.frame_count} frames → {dest.name}/")
        for path in tqdm(seg.frame_paths, desc=f"  game_{seg.game_index:02d}",
                         unit="frame", dynamic_ncols=True, smoothing=0, mininterval=1.0):
            shutil.move(str(path), str(dest / path.name))
            game_frame_set.add(path)
        created.append(str(dest))

    # Handle remaining frames (lobby / loading screens)
    remaining = [p for p in sorted(base.glob("frame_*.jpg")) if p not in game_frame_set]
    if remaining:
        if move_lobby:
            lobby = base / "lobby"
            lobby.mkdir(exist_ok=True)
            print(f"Moving {len(remaining)} lobby/loading frames → lobby/")
            for path in tqdm(remaining, desc="  lobby", unit="frame", dynamic_ncols=True,
                             smoothing=0, mininterval=1.0):
                shutil.move(str(path), str(lobby / path.name))
        else:
            print(f"Deleting {len(remaining)} lobby/loading frames")
            for path in remaining:
                path.unlink()

    return created
