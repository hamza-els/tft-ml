"""
Saving and scoring in the champion labeler (scripts/label_champions.py).

Usage:
    python -m pytest tests/test_state/test_champion_reviews.py -v
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.label_champions import merge_save, scoreboard


def test_merge_save_keeps_other_edits(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"a": "Ashe", "b": "Vi"}), encoding="utf-8")
    # another tool changed "b" after we loaded; we only change "c" and remove "a"
    f.write_text(json.dumps({"a": "Ashe", "b": "Taric"}), encoding="utf-8")
    merge_save(f, {"c": "Gnar", "a": None})
    assert json.loads(f.read_text(encoding="utf-8")) == {"b": "Taric", "c": "Gnar"}


def test_scoreboard_counts():
    reviews = {
        "x": {"pred": {"name": "Ashe", "star": 2, "items": ["B.F. Sword"]},
              "final": {"name": "Ashe", "star": 2, "items": ["B.F. Sword"]}},
        "y": {"pred": {"name": "Vi", "star": 1, "items": []},
              "final": {"name": "Taric", "star": 2, "items": ["Giant's Belt"]}},
        "z": {"pred": {"name": "Vi", "star": 1, "items": []},
              "final": {"name": "_none"}},
    }
    s = scoreboard(reviews)
    assert s["champion"] == [1, 3]
    assert s["star"] == [1, 2]          # the "not a unit" review has no star
    assert s["item count"] == [1, 2]
    assert s["item names"] == [1, 1]    # only slots both have are compared by name
