"""
Cleaning of board/bench unit readings over time (src/state/units.py).

Usage:
    python -m pytest tests/test_state/test_units.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.state.game_data import load_game_data
from src.state.units import clean_units, frame_weights, smooth


def _unit(at, name, p=0.95, star=1, items=()):
    return {"at": at, "name": name, "p": p, "star": star, "items": list(items)}


def _rows(readings, step=2):
    """One read row per entry (frames `step` apart = 1 s each)."""
    return [{"frame": i * step, "units_raw": units} for i, units in enumerate(readings)]


def test_frame_weight_is_time_until_next_frame_capped():
    # 2 fps: 1 frame apart = 0.5 s, 40 frames apart = 20 s -> capped at 10
    assert frame_weights([0, 1, 41, 43]) == [0.5, 10.0, 1.0, 0.5]


def test_one_frame_disappearance_is_filled():
    assert smooth(["A", "A", None, "A"], [1, 1, 1, 1]) == ["A", "A", "A", "A"]


def test_one_frame_misread_is_fixed():
    assert smooth(["A", "A", "B", "A", "A"], [1] * 5) == ["A"] * 5


def test_lasting_change_is_kept():
    assert smooth(["A", "A", "B", "B", "B"], [1] * 5) == ["A", "A", "B", "B", "B"]
    # a 3-frame stretch is not a flicker, even between two agreeing runs
    assert smooth(["A", "B", "B", "B", "A"], [5, 1, 1, 1, 5]) == ["A", "B", "B", "B", "A"]


def test_stable_frame_outweighs_flickers():
    # the lone B stood still on screen for 10 s; the A's around it were brief
    assert smooth(["A", "B", "A"], [0.5, 10, 0.5]) == ["A", "B", "A"]


def test_clean_units_fills_flicker_and_star():
    rows = _rows([
        [_unit("h03", "Ashe", star=2)],
        [],                                   # bar flickered
        [_unit("h03", "Ashe", star=None)],    # badge hidden
        [_unit("h03", "Ashe", star=2)],
    ])
    out, stats = clean_units(rows, [5] * 4)
    assert [len(o["board"]) for o in out] == [1, 1, 1, 1]
    assert all(o["board"][0]["name"] == "Ashe" and o["board"][0]["star"] == 2 for o in out)
    assert out[1]["board"][0]["hex"] == [0, 3]
    assert stats["units_filled"] == 1


def test_board_capped_at_level_dropping_least_sure():
    rows = _rows([[_unit("h00", "Ashe"), _unit("h01", "Jinx"), _unit("h02", "Sona", p=0.3)]] * 3)
    out, stats = clean_units(rows, [2] * 3)
    assert sorted(u["name"] for u in out[0]["board"]) == ["Ashe", "Jinx"]
    assert stats["over_level_frames"] == 3


def test_trait_units_do_not_count_against_level():
    rows = _rows([[_unit("h00", "Ashe"), _unit("h01", "Taric"), _unit("h02", "Stonebark Tree", p=0.3)]])
    out, stats = clean_units(rows, [2], load_game_data())
    assert len(out[0]["board"]) == 3 and stats["over_level_frames"] == 0


def test_team_size_item_raises_the_cap():
    rows = _rows([[_unit("h00", "Ashe", items=["Tactician's Crown"]), _unit("h01", "Taric"),
                   _unit("h02", "Sona", p=0.3)]])
    out, stats = clean_units(rows, [2], load_game_data())
    assert len(out[0]["board"]) == 3 and stats["over_level_frames"] == 0


def test_elder_dragon_takes_two_slots():
    rows = _rows([[_unit("h00", "Elder Dragon"), _unit("h01", "Taric"), _unit("h02", "Ashe", p=0.3)]])
    out, _ = clean_units(rows, [3], load_game_data())
    assert sorted(u["name"] for u in out[0]["board"]) == ["Elder Dragon", "Taric"]


def test_emblem_adds_trait():
    g = load_game_data()
    trait = g.emblems["Brawler Emblem"]
    plain = g.trait_counts(["Taric"], [{"name": "Taric", "items": []}]).get(trait, 0)
    with_emblem = g.trait_counts(["Taric"], [{"name": "Taric", "items": ["Brawler Emblem"]}]).get(trait, 0)
    assert with_emblem == plain + 1 or "Brawler" in g.champions["Taric"].traits


def test_counter_team_size_overrides_level():
    # "10/10" at level 9 (an augment): the 10th unit stays
    units = [_unit(f"h0{c}", n) for c, n in enumerate(["Ashe", "Taric", "Sona", "Gnar", "Vi", "Ivern", "Rakan"])] \
        + [_unit("h10", "Lux"), _unit("h11", "Varus"), _unit("h12", "Sivir", p=0.3)]
    rows = _rows([units] * 3)
    for r in rows:
        r["board_counter"] = [10, 10]
    out, stats = clean_units(rows, [9] * 3, load_game_data(), [(5, 1)] * 3)
    assert len(out[0]["board"]) == 10 and stats["team_size_from_counter"] == 3


def test_counter_unit_count_needs_agreement():
    units = [_unit("h00", "Ashe"), _unit("h01", "Taric"), _unit("h02", "Sona", p=0.3)]
    rows = _rows([units] * 3)
    rows[1]["board_counter"] = [2, 3]           # one frame alone: not trusted
    out, _ = clean_units(rows, [3] * 3, load_game_data(), [(3, 2)] * 3)
    assert len(out[1]["board"]) == 3
    for r in rows:
        r["board_counter"] = [2, 3]             # all agree: the least sure unit goes
    out, _ = clean_units(rows, [3] * 3, load_game_data(), [(3, 2)] * 3)
    assert sorted(u["name"] for u in out[1]["board"]) == ["Ashe", "Taric"]


def test_board_carried_through_unread_frames():
    rows = _rows([[_unit("h10", "Ashe"), _unit("s4", "Jinx")]]) + [{"frame": 10}, {"frame": 12}]
    out, _ = clean_units(rows, [3] * 3)
    assert out[2]["board"][0]["name"] == "Ashe" and not out[2]["read"]
    assert out[2]["bench"][0]["slot"] == 4
