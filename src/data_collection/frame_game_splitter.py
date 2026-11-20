"""
Split already-extracted frames into per-game subfolders using HUD detection.
Works on frames that were extracted from multi-game streams.

Frames are moved (not copied) into subfolders:
  <frame_dir>/game_00/
  <frame_dir>/game_01/
  ...

Frames outside detected games (lobby, loading screens, post-elimination
spectating) are moved to:
  <frame_dir>/lobby/

Time is derived from the frame number in the filename (frame_XXXXXX.jpg),
not from how many files exist, so this works the same before or after dedup.

How a game is found (see src/vision/hud.py for the two HUD icons):
  - The stage icon is on screen for the whole game. A game is a run of
    stage-icon frames with no gap longer than `max_gap_sec`.
  - If the bar returns to its stage-1 layout after a long stretch of the
    normal layout, a new game has started even without a gap.
  - The game is trimmed to end at the last frame showing the gold icon:
    after elimination the streamer spectates with the stage bar still up,
    but those frames carry no player state.
"""

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.vision.hud import HudDetector


@dataclass
class HudSample:
    frame_number: int
    stage_visible: bool
    shop_visible: bool
    stage_one_layout: bool


@dataclass
class FrameGameSegment:
    game_index: int
    start_frame: int   # frame numbers, inclusive
    end_frame: int
    frame_paths: list[Path]

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


def frame_number(path: Path) -> int:
    return int(re.search(r"(\d+)$", path.stem).group(1))


def scan_hud(
    paths: list[Path],
    detector: HudDetector,
    step_frames: int,
    workers: int = 8,
) -> list[HudSample]:
    """Run HUD detection on roughly one frame per `step_frames` frame numbers."""
    selected, last = [], None
    for p in paths:
        n = frame_number(p)
        if last is None or n - last >= step_frames:
            selected.append(p)
            last = n

    def sample(p: Path) -> HudSample | None:
        frame = cv2.imread(str(p))
        if frame is None:
            return None
        hud = detector.detect(frame)
        return HudSample(frame_number(p), hud.stage_visible, hud.shop_visible, hud.stage_one_layout)

    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(sample, selected), total=len(selected),
                            desc="Scanning HUD", unit="frame", dynamic_ncols=True,
                            smoothing=0, mininterval=1.0))
    return [s for s in results if s is not None]


def segment_games(
    samples: list[HudSample],
    fps: float = 2.0,
    max_gap_sec: float = 90.0,
    min_game_sec: float = 600.0,
    restart_after_sec: float = 300.0,
) -> list[tuple[int, int]]:
    """
    Turn HUD samples into (start_frame, end_frame) ranges, one per game.

    Args:
        samples: Output of scan_hud(), in frame order.
        fps: FPS the frames were extracted at (frame number → seconds).
        max_gap_sec: A stage-icon gap longer than this ends the game.
        min_game_sec: Discard games shorter than this.
        restart_after_sec: A return to the stage-1 layout after this long in
            the normal layout starts a new game.
    """
    max_gap = max_gap_sec * fps
    restart_after = restart_after_sec * fps

    runs: list[list[HudSample]] = []
    current: list[HudSample] = []
    normal_since: int | None = None   # first frame of the normal (post stage 1) layout

    for s in samples:
        if not s.stage_visible:
            continue
        gap = current and s.frame_number - current[-1].frame_number > max_gap
        restarted = (current and s.stage_one_layout and normal_since is not None
                     and s.frame_number - normal_since > restart_after)
        if gap or restarted:
            runs.append(current)
            current, normal_since = [], None
        current.append(s)
        if not s.stage_one_layout and normal_since is None:
            normal_since = s.frame_number
    if current:
        runs.append(current)

    games = []
    for run in runs:
        alive = [s.frame_number for s in run if s.shop_visible]
        if not alive:
            continue
        start, end = run[0].frame_number, alive[-1]
        if (end - start) / fps >= min_game_sec:
            games.append((start, end))
    return games


def find_game_segments(
    frame_dir: str,
    detector: HudDetector | None = None,
    extraction_fps: float = 2.0,
    scan_step_sec: float = 2.5,
    max_gap_sec: float = 90.0,
    min_game_sec: float = 600.0,
) -> list[FrameGameSegment]:
    """
    Scan extracted frames and group them into game segments.

    Args:
        frame_dir: Directory containing frame_XXXXXX.jpg files.
        detector: HUD detector (default: built from vision_config.yaml).
        extraction_fps: FPS used when frames were extracted (default 2.0).
        scan_step_sec: Time between scanned frames; sets boundary precision.
        max_gap_sec: Seconds without the stage bar that end a game.
        min_game_sec: Discard segments shorter than this.

    Returns:
        List of FrameGameSegment objects, one per detected game.
    """
    paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    detector = detector or HudDetector()
    step = max(1, int(scan_step_sec * extraction_fps))
    samples = scan_hud(paths, detector, step)
    ranges = segment_games(samples, extraction_fps, max_gap_sec, min_game_sec)

    segments = []
    for i, (start, end) in enumerate(ranges):
        # The first/last sample are up to one scan step inside the true boundary.
        lo, hi = start - step, end + step
        # ...but games can be back to back (a new game found from the stage-1 layout, no gap):
        # never let the padding overlap the neighbours, or a frame is moved twice
        # (wszJroIjlMs, 2026-11-19: game 6 ended at 28866, game 7 began at 28876).
        if i > 0:
            lo = max(lo, (ranges[i - 1][1] + start) // 2 + 1)
        if i + 1 < len(ranges):
            hi = min(hi, (end + ranges[i + 1][0]) // 2)
        frames = [p for p in paths if lo <= frame_number(p) <= hi]
        segments.append(FrameGameSegment(i, start, end, frames))
        print(f"  Game {i}: frames {start}–{end}  "
              f"({start / extraction_fps / 60:.1f}–{end / extraction_fps / 60:.1f} min, "
              f"{(end - start) / extraction_fps / 60:.0f} min long, {len(frames)} files)")

    print(f"Found {len(segments)} game(s).")
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
