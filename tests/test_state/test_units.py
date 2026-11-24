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


def test_rider_shares_the_bff_hex():
    # a champion riding the Big Furry Friend is read as "h31r" (unit_reader.rider_bars)
    rows = _rows([[_unit("h31", "Big Furry Friend"), _unit("h31r", "Teemo", star=2), _unit("h20", "Veigar")]] * 3)
    out, stats = clean_units(rows, [2] * 3, load_game_data())
    board = {u["name"]: u for u in out[1]["board"]}
    assert set(board) == {"Big Furry Friend", "Teemo", "Veigar"} and stats["over_level_frames"] == 0
    assert board["Teemo"]["hex"] == board["Big Furry Friend"]["hex"] == [3, 1]
    assert board["Teemo"]["star"] == 2


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


_TEN = [_unit(f"h0{c}", n) for c, n in enumerate(["Ashe", "Taric", "Sona", "Gnar", "Vi", "Ivern", "Rakan"])] \
    + [_unit("h10", "Lux"), _unit("h11", "Varus"), _unit("h12", "Sivir", p=0.3)]


def test_counter_team_size_overrides_level(monkeypatch):
    # "10/10" at level 9 (an augment): with the counter as the team size, the 10th unit stays
    import src.state.units as U
    monkeypatch.setattr(U, "TEAM_SIZE_FROM", "counter")
    rows = _rows([_TEN] * 3)
    for r in rows:
        r["board_counter"] = [10, 10]
    out, stats = clean_units(rows, [9] * 3, load_game_data(), [(5, 1)] * 3)
    assert len(out[0]["board"]) == 10 and stats["team_size_from_counter"] == 3


def test_team_size_augment_counts():
    # level 9 + New Recruit: 10 units fit, no counter needed; without the augment the least sure goes
    rows = _rows([_TEN] * 3)
    out, _ = clean_units(rows, [9] * 3, load_game_data(), [(5, 1)] * 3, augments=[["New Recruit"]] * 3)
    assert len(out[0]["board"]) == 10
    out, _ = clean_units(rows, [9] * 3, load_game_data(), [(5, 1)] * 3)
    assert len(out[0]["board"]) == 9 and "Sivir" not in [u["name"] for u in out[0]["board"]]


def test_team_size_item_in_the_item_column_counts():
    rows = _rows([_TEN] * 3)
    for r in rows:
        r["bench_items"] = ["Tactician's Cape", "B.F. Sword"]
    out, _ = clean_units(rows, [9] * 3, load_game_data(), [(5, 1)] * 3)
    assert len(out[0]["board"]) == 10


def test_team_size_item_stays_owned_once_seen():
    gd = load_game_data()
    # seen in the column for 3 frames of round 5-1, then not read in 5-2: still +1
    rows = _rows([_TEN] * 6)
    for r in rows[:3]:
        r["bench_items"] = ["Tactician's Crown"]
    out, _ = clean_units(rows, [9] * 6, gd, [(5, 1)] * 3 + [(5, 2)] * 3)
    assert [len(o["board"]) for o in out] == [10] * 6
    assert gd.team_size(9, [], (), ["Coronation"]) == 10      # Coronation hands out a Crown
    assert gd.team_size(8, [], (), ["Cursed Crown"]) == 10


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


def _counted(readings, count):
    rows = _rows(readings)
    for r in rows:
        r["board_counter"] = [count, 9]
    return rows


def test_hidden_unit_carried_when_counter_says_it_is_there():
    a, b, c = _unit("h00", "Ashe"), _unit("h01", "Taric"), _unit("h02", "Sona")
    # Sona's bar hidden for the last 4 frames (longer than a flicker), counter still says 3
    rows = _counted([[a, b, c]] * 3 + [[a, b]] * 4, 3)
    out, stats = clean_units(rows, [9] * 7, load_game_data(), [(4, 1)] * 7)
    assert sorted(u["name"] for u in out[-1]["board"]) == ["Ashe", "Sona", "Taric"]
    assert [u for u in out[-1]["board"] if u["name"] == "Sona"][0]["carried"]
    assert stats["units_carried"] >= 1


def test_moved_unit_not_carried_twice():
    a, b = _unit("h00", "Ashe"), _unit("h01", "Taric")
    moved = _unit("h05", "Taric")
    rows = _counted([[a, b]] * 3 + [[a, moved]] * 4, 3)   # counter misreads 3; Taric only moved
    out, _ = clean_units(rows, [9] * 7, load_game_data(), [(4, 1)] * 7)
    assert sorted(u["name"] for u in out[-1]["board"]) == ["Ashe", "Taric"]


def test_nothing_carried_without_counter_or_level():
    a, b, c = _unit("h00", "Ashe"), _unit("h01", "Taric"), _unit("h02", "Sona")
    rows = _rows([[a, b, c]] * 3 + [[a, b]] * 4)
    out, stats = clean_units(rows, [None] * 7, load_game_data(), [(4, 1)] * 7)
    assert len(out[-1]["board"]) == 2 and stats["units_carried"] == 0


def test_vanished_unit_carried_up_to_the_team_size():
    # no counter, level 3: the board should be full, so Sona (seen a moment ago) is put back
    a, b, c = _unit("h00", "Ashe"), _unit("h01", "Taric"), _unit("h02", "Sona")
    rows = _rows([[a, b, c]] * 3 + [[a, b]] * 4)
    out, stats = clean_units(rows, [3] * 7, load_game_data(), [(4, 1)] * 7)
    assert len(out[3]["board"]) == 3 and out[3]["board"][2].get("carried")
    # ...but not when it went to the bench
    rows = _rows([[a, b, c]] * 3 + [[a, b, _unit("s0", "Sona")]] * 4)
    out, _ = clean_units(rows, [3] * 7, load_game_data(), [(4, 1)] * 7)
    assert [u["name"] for u in out[3]["board"]] == ["Ashe", "Taric"]


def test_tooltip_covered_reading_is_filled_from_neighbours():
    a = _unit("h00", "Ashe", star=2)
    covered = {"at": "h00", "covered": True}
    # 3 covered frames in a row would otherwise read as a lasting empty hex
    rows = _rows([[a], [a], [covered], [covered], [covered], [a]])
    out, stats = clean_units(rows, [5] * 6)
    assert all(o["board"] and o["board"][0]["name"] == "Ashe" and o["board"][0]["star"] == 2 for o in out)
    assert stats["covered_filled"] == 3


def test_all_covered_position_stays_empty():
    rows = _rows([[{"at": "s3", "covered": True}]] * 3)
    out, _ = clean_units(rows, [5] * 3)
    assert all(not o["bench"] for o in out)


def test_board_carried_through_unread_frames():
    rows = _rows([[_unit("h10", "Ashe"), _unit("s4", "Jinx")]]) + [{"frame": 10}, {"frame": 12}]
    out, _ = clean_units(rows, [3] * 3)
    assert out[2]["board"][0]["name"] == "Ashe" and not out[2]["read"]
    assert out[2]["bench"][0]["slot"] == 4


def test_loose_unit_kept_only_when_counter_is_short():
    a, b = _unit("h00", "Ashe"), _unit("h01", "Taric")
    dragged = dict(_unit("h05", "Sona", p=0.8), loose=True)
    rows = _counted([[a, b, dragged]] * 3, 3)        # counter: 3 on the board
    out, _ = clean_units(rows, [9] * 3, load_game_data(), [(4, 1)] * 3)
    assert sorted(u["name"] for u in out[1]["board"]) == ["Ashe", "Sona", "Taric"]
    rows = _counted([[a, b, dragged]] * 3, 2)        # counter: 2, the loose one is from the bench
    out, stats = clean_units(rows, [9] * 3, load_game_data(), [(4, 1)] * 3)
    assert sorted(u["name"] for u in out[1]["board"]) == ["Ashe", "Taric"]
    assert stats["loose_dropped"] == 3


def test_sure_unit_not_erased_between_empty_frames():
    # bar hidden on most frames, but the one reading is sure: the unit was there
    rows = _rows([[], [], [_unit("h10", "Rammus", p=0.96)], [], []])
    out, _ = clean_units(rows, [5] * 5)
    assert [u["name"] for u in out[2]["board"]] == ["Rammus"]
    # an unsure lone reading between empty frames is still taken as noise
    rows = _rows([[], [], [_unit("h10", "Rammus", p=0.4)], [], []])
    out, _ = clean_units(rows, [5] * 5)
    assert out[2]["board"] == []


# --- step 6: stretches with the same gold and bench (Hamza, 2026-11-24) ---

def _stretch_rows(boards, gold, bench=("s0", "Vi")):
    """Read rows of one round: each board plus the same bench unit, with the given gold per row."""
    return [{"frame": i * 2, "units_raw": b + [_unit(*bench)]} for i, b in enumerate(boards)], gold


def test_unit_missing_longer_than_a_flicker_is_filled_when_nothing_else_changed():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx", p=0.7)
    boards = [[ashe, jinx]] * 3 + [[ashe]] * 4 + [[ashe, jinx]] * 3     # 4 frames: no flicker
    rows, gold = _stretch_rows(boards, [10] * 10)
    out, stats = clean_units(rows, [5] * 10, rounds=[(2, 1)] * 10, gold=gold)
    assert all(sorted(u["name"] for u in o["board"]) == ["Ashe", "Jinx"] for o in out)
    assert out[4]["board"][1]["hex"] == [0, 5] and out[4]["board"][1]["filled"]
    assert stats["stretch_filled"] == 4


def test_gold_change_ends_the_stretch():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx", p=0.7)
    boards = [[ashe, jinx]] * 3 + [[ashe]] * 4 + [[ashe, jinx]] * 3
    rows, _ = _stretch_rows(boards, None)
    gold = [10] * 3 + [12] * 4 + [10] * 3      # sold then bought back: two changes
    out, _ = clean_units(rows, [None] * 10, rounds=[(2, 1)] * 10, gold=gold)
    assert [len(o["board"]) for o in out] == [2] * 3 + [1] * 4 + [2] * 3


def test_bench_change_ends_the_stretch():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx", p=0.7)
    rows = [{"frame": i * 2, "units_raw": b} for i, b in enumerate(
        [[ashe, jinx]] * 3 + [[ashe, _unit("s2", "Jinx")]] * 4 + [[ashe, jinx]] * 3)]
    out, _ = clean_units(rows, [None] * 10, rounds=[(2, 1)] * 10, gold=[10] * 10)
    assert [len(o["board"]) for o in out] == [2] * 3 + [1] * 4 + [2] * 3


def test_unread_gold_links_only_when_the_readings_around_agree():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx", p=0.7)
    boards = [[ashe, jinx]] * 3 + [[ashe]] * 4 + [[ashe, jinx]] * 3
    rows, _ = _stretch_rows(boards, None)
    out, _ = clean_units(rows, [None] * 10, rounds=[(2, 1)] * 10, gold=[10] * 3 + [None] * 4 + [10] * 3)
    assert [len(o["board"]) for o in out] == [2] * 10
    out, _ = clean_units(rows, [None] * 10, rounds=[(2, 1)] * 10, gold=[10] * 3 + [None] * 4 + [13] * 3)
    assert [len(o["board"]) for o in out] == [2] * 3 + [1] * 4 + [2] * 3


def test_stretch_drops_a_lasting_misread_extra_unit():
    ashe = _unit("h03", "Ashe")
    boards = [[ashe]] * 5 + [[ashe, _unit("h12", "Lux", p=0.4)]] * 3 + [[ashe]] * 4
    rows, gold = _stretch_rows(boards, [10] * 12)
    out, stats = clean_units(rows, [5] * 12, rounds=[(2, 1)] * 12, gold=gold)
    assert all([u["name"] for u in o["board"]] == ["Ashe"] for o in out)
    assert stats["stretch_removed"] == 3


def test_stretch_gives_one_star_level():
    boards = [[_unit("h03", "Ashe", star=2)]] * 5 + [[_unit("h03", "Ashe", star=1, p=0.6)]] * 3
    rows, gold = _stretch_rows(boards, [10] * 8)
    out, _ = clean_units(rows, [5] * 8, rounds=[(2, 1)] * 8, gold=gold)
    assert [o["board"][0]["star"] for o in out] == [2] * 8


def test_stretch_takes_the_counters_count():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx", p=0.9)
    boards = [[ashe, jinx]] * 2 + [[ashe]] * 6                  # Jinx mostly hidden: the vote alone drops it
    rows, gold = _stretch_rows(boards, [10] * 8)
    for r in rows:
        r["board_counter"] = [2, 5]
    out, _ = clean_units(rows, [5] * 8, rounds=[(2, 1)] * 8, gold=gold)
    assert [len(o["board"]) for o in out] == [2] * 8


def test_unit_dragged_over_another_goes_back_to_its_hex():
    ashe, jinx = _unit("h03", "Ashe"), _unit("h05", "Jinx")
    # frames 3-5 (longer than a flicker): Jinx's bar gone from h05, h03 read as Jinx
    # (dragged over Ashe), Ashe missing
    boards = [[ashe, jinx]] * 3 + [[_unit("h03", "Jinx", p=0.5)]] * 3 + [[ashe, jinx]] * 3
    rows, gold = _stretch_rows(boards, [10] * 9)
    out, stats = clean_units(rows, [5] * 9, rounds=[(2, 1)] * 9, gold=gold)
    for o in out[3:6]:
        assert {(u["name"], tuple(u["hex"])) for u in o["board"]} == {("Ashe", (0, 3)), ("Jinx", (0, 5))}
    assert stats["stretch_stacked"] == 3


from src.state.units import accounted, free_unit_round


def test_free_unit_rounds():
    assert free_unit_round((1, 3)) and free_unit_round((3, 4)) and free_unit_round((4, 7))
    assert free_unit_round((2, 1)) and free_unit_round((4, 2)) and free_unit_round(None)
    assert not free_unit_round((3, 3)) and not free_unit_round((5, 1))


def test_accounted_champions():
    rows = [{"shop_slots": ["Ashe", None], "units_raw": []},                   # 3-3: Ashe in the shop
            {"shop_slots": [], "units_raw": [_unit("h00", "Vi", p=0.95)]},     # 3-3: Vi read surely
            {"shop_slots": [], "units_raw": [_unit("h00", "Lux", p=0.6)] * 2},  # 3-4 carousel: Lux arrives
            {"shop_slots": [], "units_raw": []},                               # 3-5
            {"shop_slots": [], "units_raw": []}]                               # 3-6
    ok = accounted(rows, [(3, 3), (3, 3), (3, 4), (3, 5), (3, 6)], {"_unknown"})
    assert {"Ashe", "Vi", "_unknown"} <= ok[0] and ok[2] is None
    assert "Lux" in ok[3] and "Ashe" not in ok[3]          # Lux from the carousel; Ashe's shop is 2 rounds back
    assert ok[4] == {"_unknown"}


def test_doubted_reading_takes_a_next_guess_that_had_a_way_there():
    rows = [{"frame": 0, "shop_slots": ["Ashe"], "units_raw": []},
            {"frame": 2, "units_raw": [dict(_unit("h03", "Jinx", p=0.6), alt=[["Ashe", 0.3], ["Vi", 0.05]])]},
            {"frame": 4, "units_raw": [_unit("h04", "Lux", p=0.8)]}]
    out, stats = clean_units(rows, [5] * 3, rounds=[(3, 3)] * 3)
    assert [u["name"] for u in out[1]["board"]] == ["Ashe"]
    lux = next(u for u in out[2]["board"] if u["name"] == "Lux")
    assert lux["doubted"] and lux["p"] == 0.4
    assert stats["doubted_renamed"] == 1 and stats["doubted"] == 1
