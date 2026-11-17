"""
Tests for the YOLO labeling/training pipeline and shop OCR name snapping.
All synthetic — no real frames or trained weights needed.

Usage:
    python -m pytest tests/test_vision/test_labeling_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest
import yaml

from src.data_collection.label_sampler import find_sources, labeled_name, pick_evenly, sample_frames
from src.vision.champion_detector import _infer_region
from src.vision.ocr_pipeline import snap_name
from src.vision.yolo_dataset import build_dataset, group_split, labeled_pairs, source_of

class _Hud:
    def __init__(self, shop_visible: bool):
        self.shop_visible = shop_visible


class _FakeDetector:
    """Stands in for HudDetector: a noise frame has the bottom bar, a black frame doesn't."""
    def detect(self, frame):
        return _Hud(bool(frame.mean() > 40))


HUD = _FakeDetector()


def _write_frame(path: Path, in_game: bool):
    """Noise frame = HUD active; black frame = lobby/loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(str(path)))
    img = (rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) if in_game
           else np.zeros((1080, 1920, 3), dtype=np.uint8))
    cv2.imwrite(str(path), img)


# ── label_sampler ─────────────────────────────────────────────────────────────

def test_pick_evenly_spreads_and_keeps_ends():
    assert pick_evenly(list(range(100)), 5) == [0, 25, 50, 74, 99]


def test_pick_evenly_returns_all_when_short():
    assert pick_evenly([1, 2, 3], 10) == [1, 2, 3]
    assert pick_evenly([1, 2, 3], 0) == []


def test_find_sources_skips_lobby(tmp_path):
    for rel in ("vodA/frame_000001.jpg",
                "vodB/game_00/frame_000001.jpg",
                "vodB/lobby/frame_000002.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    sources = [s.relative_to(tmp_path).as_posix() for s in find_sources(tmp_path)]
    assert sources == ["vodA", "vodB/game_00"]


def test_labeled_name_keeps_vod_prefix(tmp_path):
    p = tmp_path / "z5w_IgSE3K38" / "game_01" / "frame_000123.jpg"
    name = labeled_name(p, tmp_path)
    assert name == "z5w_IgSE3K38__game_01__frame_000123.jpg"
    assert source_of(Path(name)) == "z5w_IgSE3K38"


def test_sample_frames_skips_lobby_and_is_idempotent(tmp_path):
    root, out = tmp_path / "frames", tmp_path / "labeled"
    for i in range(10):
        _write_frame(root / "vodA" / f"frame_{i:06d}.jpg", in_game=i >= 4)

    copied = sample_frames(str(root), str(out), total=3, hud=HUD)
    assert len(copied) == 3
    assert all(int(p.stem.split("_")[-1]) >= 4 for p in copied)

    assert sample_frames(str(root), str(out), total=3, hud=HUD) == []


# ── yolo_dataset ──────────────────────────────────────────────────────────────

def _pairs(counts: dict[str, int]) -> list[tuple[Path, Path]]:
    return [(Path(f"{vod}__frame_{i:06d}.jpg"), Path(f"{vod}__frame_{i:06d}.txt"))
            for vod, n in counts.items() for i in range(n)]


def test_group_split_never_shares_a_vod():
    train, val = group_split(_pairs({"a": 40, "b": 30, "c": 20, "d": 10}), val_frac=0.2)
    train_vods = {source_of(i) for i, _ in train}
    val_vods = {source_of(i) for i, _ in val}
    assert val_vods and train_vods
    assert not train_vods & val_vods
    assert len(train) + len(val) == 100


def test_group_split_single_vod_holds_out_whole_games():
    pairs = [(Path(f"vod__game_{g:02d}__frame_{i:06d}.jpg"), Path("x.txt"))
             for g in range(5) for i in range(10)]
    train, val = group_split(pairs, val_frac=0.2)
    games = lambda items: {i.stem.split("__")[1] for i, _ in items}
    assert val and not games(train) & games(val)


def test_group_split_single_vod_falls_back_to_frames():
    train, val = group_split(_pairs({"a": 20}), val_frac=0.25)
    assert len(val) == 5 and len(train) == 15


def test_build_dataset(tmp_path):
    labeled = tmp_path / "labeled"
    (labeled / "images").mkdir(parents=True)
    (labeled / "labels").mkdir()
    (labeled / "classes.txt").write_text("Jinx\nVi\n")
    for vod in ("a", "b", "c"):
        for i in range(4):
            stem = f"{vod}__frame_{i:06d}"
            (labeled / "images" / f"{stem}.jpg").write_bytes(b"x")
            (labeled / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (labeled / "images" / "a__frame_000099.jpg").write_bytes(b"x")  # not labeled yet

    assert len(labeled_pairs(labeled)) == 12

    data_yaml = build_dataset(str(labeled), str(tmp_path / "yolo"), val_frac=0.3)
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["names"] == {0: "Jinx", 1: "Vi"}

    n_train = len(list((tmp_path / "yolo" / "train" / "images").iterdir()))
    n_val = len(list((tmp_path / "yolo" / "val" / "images").iterdir()))
    assert n_train + n_val == 12 and n_val > 0
    assert len(list((tmp_path / "yolo" / "val" / "labels").iterdir())) == n_val


def test_build_dataset_requires_classes(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        build_dataset(str(tmp_path), str(tmp_path / "yolo"))


# ── champion_detector / shop OCR ──────────────────────────────────────────────

@pytest.mark.parametrize("y, expected", [(300, "board"), (750, "bench"), (1000, None)])
def test_infer_region(y, expected):
    assert _infer_region(y, 1080) == expected


@pytest.mark.parametrize("text, expected", [
    ("Jinx", "Jinx"),
    ("jlnx", "Jinx"),
    ("Cho Gath", "Cho'Gath"),
    ("Dr.  Mundo", "Dr. Mundo"),
    ("", None),
    ("zzzzzz", None),
])
def test_snap_name(text, expected):
    assert snap_name(text, ["Jinx", "Vi", "Cho'Gath", "Dr. Mundo"]) == expected


def test_snap_name_without_list_returns_cleaned_text():
    assert snap_name("  Dr.   Mundo ", []) == "Dr. Mundo"
