"""
Star level and items read off units' health bars (src/vision/unit_bar.py).

Fixtures are small crops of real frames (no webcam): the area around one
board unit's health bar, pasted back at its original position on a black
1920x1080 frame so the reader sees real screen coordinates.

Usage:
    python -m pytest tests/test_vision/test_unit_bar.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

from src.vision.board import BoardHexes
from src.vision.unit_bar import UnitItemReader, board_bar, star_level

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# fixture, hex (row, col), where the crop came from (x, y), stars, items
UNITS = [
    ("unit_1star_no_items.png", (3, 5), (1146, 485), 1, []),
    ("unit_2star_3_items.png", (0, 2), (702, 269), 2, ["Steadfast Heart", None, "Protector's Vow"]),
    ("unit_3star_3_items.png", (2, 6), (1199, 416), 3, ["Sprykin Emblem", "Fae Emblem", "Edge of Night"]),
]


def _frame(name: str, at: tuple[int, int]) -> np.ndarray:
    crop = cv2.imread(str(FIXTURES / name))
    assert crop is not None, f"fixture missing: {name}"
    frame = np.zeros((1080, 1920, 3), np.uint8)
    x, y = at
    frame[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return frame


@pytest.fixture(scope="module")
def reader():
    return UnitItemReader()


@pytest.mark.parametrize("name, hex_, at, stars, items", UNITS)
def test_star_level_and_items(reader, name, hex_, at, stars, items):
    frame = _frame(name, at)
    bar = board_bar(frame, BoardHexes(), *hex_)
    assert bar is not None
    assert star_level(frame, bar) == stars
    got = reader.read(frame, bar)
    assert len(got) == len(items)
    for want, have in zip(items, got):
        if want is not None:            # None: an item whose exact template may change on re-harvest
            assert have == want


def test_no_bar_on_empty_hex():
    frame = _frame("unit_1star_no_items.png", (1146, 485))
    assert board_bar(frame, BoardHexes(), 3, 2) is None


def test_neighbour_bar_is_not_taken():
    # The unit stands on hex (3, 5); the hex next to it must not claim its bar.
    frame = _frame("unit_1star_no_items.png", (1146, 485))
    assert board_bar(frame, BoardHexes(), 3, 4) is None
    assert board_bar(frame, BoardHexes(), 3, 6) is None
