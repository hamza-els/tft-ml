"""
Augment bar: counts the stage allows, and whose board a frame shows when both bars are up.

Usage:
    python -m pytest tests/test_vision/test_augment_counts.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vision.arena import home_board_rows
from src.vision.augments import MAX_AUGMENTS, expected_counts


def test_expected_counts():
    assert expected_counts(None, 3) is None
    assert expected_counts(2, 1) == set(range(1, MAX_AUGMENTS + 1))   # the first pick is happening
    assert expected_counts(3, 2) == set(range(1, MAX_AUGMENTS + 1))   # 2-1 done; 3-2 may not be yet
    assert expected_counts(3, 3) == set(range(2, MAX_AUGMENTS + 1))
    assert expected_counts(5, 1) == set(range(3, MAX_AUGMENTS + 1))   # an extra augment can make 4


def test_both_bars_fall_back_to_colour():
    # No arena signatures: the colour fallback says "home".
    rows = [{"augments": ["a"], "away_augments": 0},     # own bar only: home
            {"augments": [], "away_augments": 2},        # other player's bar only: away
            {"augments": ["a"], "away_augments": 2},     # combat, both bars: colour decides
            {"augments": [], "away_augments": 0}]        # no bar: colour decides
    assert home_board_rows(rows) == [True, False, True, True]
