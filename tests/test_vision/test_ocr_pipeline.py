"""
OCR pipeline tests against a real extracted frame.

Usage:
    python -m pytest tests/test_vision/test_ocr_pipeline.py -v -s
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import pytest

from src.vision.ocr_pipeline import extract_ui_values

# ── Sample frame ──────────────────────────────────────────────────────────────
SAMPLE_FRAME = Path(r"C:\Users\hamza\tft_frames_temp\z5wIgSE3K38\frame_006383.jpg")


@pytest.fixture(scope="session")
def frame():
    assert SAMPLE_FRAME.exists(), (
        f"Sample frame not found: {SAMPLE_FRAME}\n"
        "Update SAMPLE_FRAME to any extracted frame path."
    )
    img = cv2.imread(str(SAMPLE_FRAME))
    assert img is not None, "cv2.imread returned None — file may be corrupt"
    return img


@pytest.fixture(scope="session")
def ocr_results(frame):
    """Run OCR once for the whole session — EasyOCR is slow to initialise."""
    return extract_ui_values(frame)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ocr_runs_without_error(frame):
    result = extract_ui_values(frame)
    assert isinstance(result, dict)


def test_ocr_returns_expected_keys(ocr_results):
    expected = {"gold", "hp", "level", "stage", "round", "streak", "xp_current", "xp_needed"}
    assert expected == set(ocr_results.keys())


def test_gold(ocr_results):
    gold = ocr_results["gold"]
    print(f"\n  gold = {gold}")
    assert gold is not None, "Gold OCR returned None — check region in vision_config.yaml"
    assert 0 <= gold <= 100, f"Gold out of range: {gold}"
    assert gold == int(gold), f"Gold should be a whole number, got {gold}"


def test_hp(ocr_results):
    hp = ocr_results["hp"]
    print(f"\n  hp = {hp}")
    assert hp is not None, "HP OCR returned None — check region in vision_config.yaml"
    assert 0 < hp <= 100, f"HP out of range: {hp}"


def test_level(ocr_results):
    level = ocr_results["level"]
    print(f"\n  level = {level}")
    assert level is not None, "Level OCR returned None — check region in vision_config.yaml"
    assert 1 <= level <= 9, f"Level out of range: {level}"
    assert level == int(level), f"Level should be a whole number, got {level}"


def test_stage(ocr_results):
    stage = ocr_results["stage"]
    round_ = ocr_results["round"]
    print(f"\n  stage = {stage}  round = {round_}")
    assert stage is not None, "Stage OCR returned None — check region in vision_config.yaml"
    assert 1 <= stage <= 7, f"Stage out of range: {stage}"
    assert round_ is not None, "Round OCR returned None — stage text may not contain '-'"
    assert 1 <= round_ <= 7, f"Round out of range: {round_}"


def test_streak(ocr_results):
    streak = ocr_results["streak"]
    print(f"\n  streak = {streak}")
    if streak is not None:
        assert 0 <= streak <= 15, f"Streak out of range: {streak}"


def test_xp(ocr_results):
    xp_cur = ocr_results["xp_current"]
    xp_need = ocr_results["xp_needed"]
    print(f"\n  xp = {xp_cur} / {xp_need}")
    if xp_cur is not None and xp_need is not None:
        assert 0 <= xp_cur <= xp_need, f"XP current > needed: {xp_cur} / {xp_need}"
        assert xp_need <= 84, f"XP needed out of range: {xp_need}"


def test_print_all_results(ocr_results):
    print("\n  ── OCR results ──────────────────")
    for k, v in ocr_results.items():
        print(f"    {k:<12} {v}")
    print("  ─────────────────────────────────")
