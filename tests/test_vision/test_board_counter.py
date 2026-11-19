"""
The board counter "units / team size" (src/vision/board_counter.py).

Fixtures are crops of real frames around the counter (no webcam), pasted
back at their position on a black 1920x1080 frame.

Usage:
    python -m pytest tests/test_vision/test_board_counter.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest

from src.vision.board_counter import CFG, BoardCounter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _frame(name: str) -> np.ndarray:
    crop = cv2.imread(str(FIXTURES / name))
    assert crop is not None, f"fixture missing: {name}"
    x, y = CFG["strip"][0] - 10, CFG["strip"][1] - 10
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return frame


@pytest.mark.parametrize("name, units, team_size", [
    ("counter_7_of_8.png", 7, 8),        # bright blue: a free slot
    ("counter_10_of_10.png", None, 10),  # level 9 + an augment; the "10" on the left is behind a nameplate
])
def test_reads_counter(name, units, team_size):
    assert BoardCounter().read(_frame(name)) == (units, team_size)


def test_nothing_read_on_empty_frame():
    assert BoardCounter().read(np.zeros((1080, 1920, 3), np.uint8)) == (None, None)
