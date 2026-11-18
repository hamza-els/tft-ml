"""
OCR pipeline tests against two real frames kept in tests/fixtures/.

  frame_030059.jpg — mid game (stage 5-6), shop open, normal stage-bar layout
  frame_010313.jpg — stage 1-3, stage bar in its shifted stage-1 position

Both are from the HUD layout configs/vision_config.yaml is calibrated for.
If these fail after recalibrating for a different stream, replace the
fixtures and expected values with frames from that stream.

Usage:
    python -m pytest tests/test_vision/test_ocr_pipeline.py -v -s
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import pytest

from src.vision.hud import HudDetector
from src.vision.ocr_pipeline import extract_ui_values, pick_hp, read_hp, read_shop

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str):
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None, f"fixture missing: {name}"
    return img


@pytest.fixture(scope="session")
def mid_game():
    return _load("frame_030059.jpg")


@pytest.fixture(scope="session")
def stage_one():
    return _load("frame_010313.jpg")


# ── HUD detection ─────────────────────────────────────────────────────────────

def test_hud_found_in_both_layouts(mid_game, stage_one):
    hud = HudDetector()
    mid, early = hud.detect(mid_game), hud.detect(stage_one)
    assert mid.stage_visible and mid.shop_visible and not mid.stage_one_layout
    assert early.stage_visible and early.shop_visible and early.stage_one_layout


def test_hud_absent_on_blank_frame(mid_game):
    blank = mid_game * 0
    state = HudDetector().detect(blank)
    assert not state.stage_visible and not state.shop_visible


# ── Scalar OCR ────────────────────────────────────────────────────────────────

def test_ui_values_mid_game(mid_game):
    values = extract_ui_values(mid_game)
    print(f"\n  {values}")
    assert values == {
        "gold": 1, "hp": 51, "level": 9, "stage": 5, "round": 6,
        "streak": 4, "xp_current": 2, "xp_needed": 68,
    }


def test_ui_values_stage_one_layout(stage_one):
    values = extract_ui_values(stage_one)
    print(f"\n  {values}")
    assert values == {
        "gold": 0, "hp": 100, "level": 2, "stage": 1, "round": 3,
        "streak": 0, "xp_current": 0, "xp_needed": 2,
    }


def test_ui_values_blank_frame_are_none(mid_game):
    values = extract_ui_values(mid_game * 0)
    assert set(values.values()) == {None}


# ── HP ────────────────────────────────────────────────────────────────────────

def test_read_hp_other_players(stage_one):
    own, others = read_hp(stage_one)
    assert own == 100
    assert others == [100.0] * 7


def test_pick_hp_rejects_junk_sharing_a_row():
    boxes = [
        ((1782, 189, 30, 18), "82", 0.69),   # clipped player name on another player's row
        ((1832, 193, 20, 12), "26", 0.73),
        ((1791, 433, 30, 22), "23", 1.0),    # own HP: large, no small number on its row
        ((1832, 523, 20, 13), "20", 0.99),
        ((1770, 601, 30, 16), "510", 0.19),  # junk: not a valid HP, low confidence
    ]
    assert pick_hp(boxes) == (23.0, [26.0, 20.0])


def test_pick_hp_none_when_own_row_missing():
    assert pick_hp([((1832, 193, 20, 12), "26", 0.9)]) == (None, [26.0])


# ── Shop ──────────────────────────────────────────────────────────────────────

def test_read_shop(mid_game, stage_one):
    assert read_shop(mid_game) == ["Morgana", "Malphite", "Shen", "Draven", None]
    assert read_shop(stage_one) == [None, None, "Ornn", "Cinderling", "Pebbles"]


# ── Augment bar ───────────────────────────────────────────────────────────────

def test_augment_bar(mid_game, stage_one):
    from src.vision.augments import AugmentReader
    reader = AugmentReader()
    assert reader.count(mid_game) == 3
    assert [a.tier for a in reader.read(mid_game)] == ["gold", "gold", "gold"]
    assert reader.count(stage_one) == 0          # no augment picked yet in stage 1
    assert reader.count(mid_game * 0) == 0       # an all-black frame is not a bar



def test_bottom_bar_shifted_left_is_followed(mid_game):
    """Later VODs draw the bottom bar ~105 px further left; values must still read the same."""
    import numpy as np
    shifted = mid_game.copy()
    shifted[860:1080, :-105] = mid_game[860:1080, 105:]
    hud = HudDetector().detect(shifted)
    assert hud.shop_visible and hud.bottom_dx == -105
    values = extract_ui_values(shifted)
    assert (values["gold"], values["level"], values["xp_current"], values["streak"]) == (1, 9, 2, 4)
    assert read_shop(shifted) == ["Morgana", "Malphite", "Shen", "Draven", None]
