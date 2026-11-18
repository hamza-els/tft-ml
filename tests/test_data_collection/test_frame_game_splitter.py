"""
Tests for game segmentation and the HUD-aware / keep-alive dedup options.
Synthetic HUD samples and frames — no real footage needed.

Usage:
    python -m pytest tests/test_data_collection/test_frame_game_splitter.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import cv2
import numpy as np

from src.data_collection.frame_dedup import WATCH_REGIONS, deduplicate_frames, frame_folders
from src.data_collection.frame_game_splitter import (
    FrameGameSegment,
    HudSample,
    frame_number,
    segment_games,
    split_into_subfolders,
)

FPS = 2.0
STEP = 5   # one sample every 2.5 s


def _game(start: int, minutes: float, stage_one_minutes: float = 2.0,
          spectate_minutes: float = 0.0) -> list[HudSample]:
    """Samples for one game: stage 1, then normal layout, then optional spectating."""
    n = int(minutes * 60 * FPS)
    s1 = int(stage_one_minutes * 60 * FPS)
    samples = [HudSample(start + i, True, True, i < s1) for i in range(0, n, STEP)]
    spec = int(spectate_minutes * 60 * FPS)
    samples += [HudSample(start + n + i, True, False, False) for i in range(0, spec, STEP)]
    return samples


def _lobby(start: int, minutes: float) -> list[HudSample]:
    return [HudSample(start + i, False, False, False)
            for i in range(0, int(minutes * 60 * FPS), STEP)]


# ── segment_games ─────────────────────────────────────────────────────────────

def test_two_games_separated_by_lobby():
    samples = _game(1000, 30) + _lobby(4600, 5) + _game(5200, 25)
    games = segment_games(samples, FPS)
    assert len(games) == 2
    assert games[0] == (1000, 1000 + 3600 - STEP)
    assert games[1][0] == 5200


def test_spectating_after_elimination_is_trimmed():
    samples = _game(0, 20, spectate_minutes=8)
    (start, end), = segment_games(samples, FPS)
    assert start == 0
    assert end == 20 * 60 * FPS - STEP      # last frame with the bottom bar


def test_short_hud_gap_inside_a_game_does_not_split():
    game = _game(0, 30)
    # drop one minute of samples mid-game (e.g. a scene switch)
    game = [s for s in game if not (1800 <= s.frame_number < 1920)]
    assert len(segment_games(game, FPS)) == 1


def test_back_to_back_games_split_on_stage_one_layout():
    # second game queues instantly: no HUD gap long enough to split on
    samples = _game(0, 30) + _game(3600 + 60, 30)
    games = segment_games(samples, FPS)
    assert [g[0] for g in games] == [0, 3660]


def test_short_segments_are_dropped():
    samples = _game(0, 4) + _lobby(480, 5) + _game(1100, 30)
    games = segment_games(samples, FPS)
    assert [g[0] for g in games] == [1100]


def test_no_hud_means_no_games():
    assert segment_games(_lobby(0, 30), FPS) == []


# ── file handling ─────────────────────────────────────────────────────────────

def test_frame_number():
    assert frame_number(Path("x/frame_030059.jpg")) == 30059


def test_split_into_subfolders_moves_games_and_lobby(tmp_path):
    for n in range(1, 11):
        (tmp_path / f"frame_{n:06d}.jpg").touch()
    game = [tmp_path / f"frame_{n:06d}.jpg" for n in range(4, 8)]
    split_into_subfolders(str(tmp_path), [FrameGameSegment(0, 4, 7, game)])

    assert sorted(p.name for p in (tmp_path / "game_00").iterdir()) == [
        f"frame_{n:06d}.jpg" for n in range(4, 8)]
    assert len(list((tmp_path / "lobby").iterdir())) == 6
    assert not list(tmp_path.glob("frame_*.jpg"))
    assert [f.name for f in frame_folders(str(tmp_path))] == ["game_00"]


# ── dedup options ─────────────────────────────────────────────────────────────

@dataclass
class _Hud:
    shop_visible: bool
    bottom_dx: int = 0


class _FakeDetector:
    """Bottom bar counts as visible when the frame's top-left pixel is white."""
    def detect(self, frame):
        return _Hud(shop_visible=bool(frame[0, 0, 0] > 200))


def _write(path: Path, bar: bool, fills: dict[str, int] | None = None):
    img = np.zeros((1080, 1920, 3), np.uint8)
    if bar:
        img[0:4, 0:4] = 255
    for name, value in (fills or {}).items():
        x, y, w, h = WATCH_REGIONS[name]
        img[y:y + h, x:x + w] = value
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def _names(folder: Path) -> list[int]:
    return [frame_number(p) for p in sorted(folder.glob("frame_*.jpg"))]


def test_keep_alive_keeps_one_frame_per_gap(tmp_path):
    for n in range(1, 31):                       # 30 identical frames = 15 s at 2 fps
        _write(tmp_path / f"frame_{n:06d}.jpg", bar=True)
    deduplicate_frames(str(tmp_path), quiet=True, max_gap_sec=5.0, fps=FPS)
    assert _names(tmp_path) == [1, 11, 21]


def test_without_keep_alive_identical_frames_collapse(tmp_path):
    for n in range(1, 31):
        _write(tmp_path / f"frame_{n:06d}.jpg", bar=True)
    deduplicate_frames(str(tmp_path), quiet=True)
    assert _names(tmp_path) == [1]


def test_hud_aware_ignores_scene_motion_while_bar_hidden(tmp_path):
    # bar hidden (carousel): bench/shop regions change every frame, stage changes once
    _write(tmp_path / "frame_000001.jpg", bar=True)
    for n, shade in zip(range(2, 7), (40, 90, 140, 190, 240)):
        fills = {"bench": shade, "shop": shade}
        if n >= 5:
            fills["stage"] = 200                 # next round starts
        _write(tmp_path / f"frame_{n:06d}.jpg", bar=False, fills=fills)
    _write(tmp_path / "frame_000007.jpg", bar=True)

    deduplicate_frames(str(tmp_path), quiet=True, hud=_FakeDetector())
    # 1: first | 2: bar disappears | 5: stage changed | 7: bar back
    assert _names(tmp_path) == [1, 2, 5, 7]


def test_hud_aware_still_sees_actions_while_bar_visible(tmp_path):
    _write(tmp_path / "frame_000001.jpg", bar=True)
    _write(tmp_path / "frame_000002.jpg", bar=True)
    _write(tmp_path / "frame_000003.jpg", bar=True, fills={"gold": 120})
    deduplicate_frames(str(tmp_path), quiet=True, hud=_FakeDetector())
    assert _names(tmp_path) == [1, 3]
