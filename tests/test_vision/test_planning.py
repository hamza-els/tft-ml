"""
Planning-phase rows and the frame classifier's veto (src/vision/arena.py).

Usage:
    python -m pytest tests/test_vision/test_planning.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vision.arena import combat_vetoed, planning_rows

CFG = {"timer_refill": 0.3, "timer_empty": 0.02, "combat_veto": 0.3}


def _row(fill, combat_p=None, counter=None):
    return {"stage": 3, "round_num": 2, "timer_fill": fill, "planning": True,
            "combat_p": combat_p, "board_counter": counter}


def test_veto_needs_the_model_and_no_counter():
    assert combat_vetoed(0.9, None, CFG)
    assert combat_vetoed(0.9, [None, None], CFG)
    assert not combat_vetoed(0.9, [5, 7], CFG)       # the counter is only drawn in planning
    assert not combat_vetoed(0.1, None, CFG)
    assert not combat_vetoed(None, None, CFG)        # model not run: no veto


def test_planning_rows_applies_the_veto():
    rows = [_row(0.8), _row(0.7, combat_p=0.95), _row(0.6, combat_p=0.95, counter=[6, 7]), _row(0.5, combat_p=0.1)]
    assert planning_rows(rows, CFG) == [True, False, True, True]
