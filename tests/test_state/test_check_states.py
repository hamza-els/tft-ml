"""
Field list, sampling and report of the state checker (scripts/check_states.py); no window opened.

Usage:
    python -m pytest tests/test_state/test_check_states.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check_states import fields_of, group_of, raw_at, report, sample

ROW = {
    "frame": 100, "stage": 3.0, "round_num": 2.0, "gold": 23.0, "hp": 88.0, "level": 6.0,
    "xp_current": 6.0, "xp_needed": 20.0, "streak": -1.0, "planning": True, "home_board": True,
    "shop_slots": [None, "Pebbles", "Kobuko", None, "Lux"], "bench_items": ["Item Remover"],
    "augments": ["Silver Destiny", "Iron Assets"], "augment_tiers": ["silver", "unknown"],
    "other_hp": [100.0, 95.0],
    "board": [{"name": "Leona", "star": 2, "items": ["Sunfire Cape"], "hex": [0, 4], "p": 0.96},
              {"name": "Rengar", "star": 1, "items": [], "hex": [1, 6], "p": 0.9, "carried": True}],
    "bench": [{"name": "Karma", "star": 1, "items": [], "slot": 3, "p": 0.95}],
}


def test_fields_cover_every_cleaned_value():
    f = {k: (g, label, v) for k, g, label, v in fields_of(ROW)}
    assert f["stage"][2] == "3-2" and f["xp"][2] == "6/20" and f["streak"][2] == "-1"
    assert f["shop.1"][2] == "Pebbles" and f["shop.0"][2] == "empty / not read"
    assert f["augment.0"][2] == "Silver Destiny (silver)" and f["augment.1"][2] == "Iron Assets"
    assert f["board.h04.name"][2] == "Leona" and f["board.h04.star"][2] == "2 star"
    assert f["board.h04.items"][2] == "Sunfire Cape" and f["bench.s3.items"][2] == "no items"
    assert "carried" in f["board.h16.name"][1]               # flags shown in the unit's label
    assert f["lobby_hp"][2] == "100 95"
    assert len([k for k in f if k.startswith("board.")]) == 8   # 4 chips per unit


def test_group_of_matches_fields_of():
    for key, group, _, _ in fields_of(ROW):
        assert group_of(key) == group


def test_report_counts_wrong_and_leaves_out_unsure():
    keys = [k for k, _, _, _ in fields_of(ROW)]
    checks = {
        "a/game_00/1": {"shown": keys, "wrong": {"gold": {"value": "23", "note": "28"},
                                                 "board.h04.star": {"value": "2 star", "note": "3"}},
                        "unsure": ["hp"]},
        "a/game_00/2": {"shown": keys, "wrong": {}, "unsure": []},
    }
    rep = {g: (n, w) for g, n, w in report(checks)}
    assert rep["gold"] == (2, 1)
    assert rep["hp"] == (1, 0)                                # the unsure one left out
    assert rep["unit star"] == (6, 1)                         # 3 units x 2 frames
    assert rep["unit there"] == (6, 0)


def test_sample_spreads_over_games_and_skips_checked():
    rows = {("v", "game_00"): [{"frame": i, "planning": i % 2 == 0} for i in range(50)],
            ("v", "game_01"): [{"frame": i, "planning": True} for i in range(100, 150)]}
    done = {f"v/game_00/{i}" for i in range(10)}
    q = sample(rows, 20, done, planning_only=False, seed=1)
    assert len(q) == 20 and len(set(q)) == 20
    assert sum(1 for _, g, _ in q if g == "game_00") == 10      # even split
    assert not any(f"{v}/{g}/{f}" in done for v, g, f in q)
    q = sample(rows, 100, set(), planning_only=True, seed=1)
    assert all(f % 2 == 0 for _, g, f in q if g == "game_00")


def test_raw_hint():
    raw = {"units_raw": [{"at": "h04", "name": "Leona", "p": 0.93}, {"at": "s3", "covered": True},
                         {"at": "h16", "name": "Rengar", "p": 0.81, "loose": True}]}
    assert raw_at(raw, "board", "h04") == "frame: Leona 0.93"
    assert raw_at(raw, "bench", "s3") == "frame: under a tooltip"
    assert raw_at(raw, "board", "h16") == "frame: Rengar 0.81 loose"
    assert raw_at(raw, "board", "h00") == "frame: nothing here"
    assert raw_at({"units_raw": None}, "board", "h00") == "frame not read"
