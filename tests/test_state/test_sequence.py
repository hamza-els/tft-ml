"""
Tests for state-sequence cleaning.

Usage:
    python -m pytest tests/test_state/test_sequence.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.state.sequence import SCALARS, clean_game, quality_report


def _rows(**columns) -> list[dict]:
    n = len(next(iter(columns.values())))
    rows = []
    for i in range(n):
        row = {"frame": i, "in_game": True, "shop_visible": True, **dict.fromkeys(SCALARS)}
        row.update({k: v[i] for k, v in columns.items()})
        rows.append(row)
    return rows


def _col(rows, key):
    return [r[key] for r in rows]


def test_stage_misread_is_replaced():
    rows = _rows(stage=[2, 2, 2, 8, 2, 2, 3, 3, 3], round_num=[1, 1, 1, 1, 1, 2, 1, 1, 1])
    cleaned = clean_game(rows)
    assert _col(cleaned, "stage") == [2, 2, 2, 2, 2, 2, 3, 3, 3]
    assert "stage" in cleaned[3]["filled"]


def test_stage_never_goes_backwards():
    rows = _rows(stage=[3, 3, 3, 3, 2, 2, 2, 3, 3], round_num=[5] * 9)
    assert _col(clean_game(rows), "stage") == [3] * 9


def test_hp_blip_dropped_and_gaps_filled():
    rows = _rows(hp=[100, 100, 100, 6, None, 6, 100, 100, 100, 90, 90, None, 90])
    assert _col(clean_game(rows), "hp") == [100] * 9 + [90] * 4


def test_hp_can_go_up_when_the_new_value_holds():
    # a heal: HP rises and stays there, so it is not a misread
    rows = _rows(hp=[40, 40, 40, 55, 55, 55, 55, 47, 47, 47])
    assert _col(clean_game(rows), "hp") == [40, 40, 40, 55, 55, 55, 55, 47, 47, 47]


def test_hp_falling_step_by_step_is_kept():
    # the damage count-down animation shows each intermediate value for one frame
    rows = _rows(hp=[80, 80, 76, 71, 69, 69, 69])
    assert _col(clean_game(rows), "hp") == [80, 80, 76, 71, 69, 69, 69]


def test_level_never_goes_down():
    rows = _rows(level=[4, 4, 4, 5, 5, 5, 3, 5, 5, 6, 6, 6])
    assert _col(clean_game(rows), "level") == [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6]


def test_gold_spike_removed_but_real_changes_kept():
    rows = _rows(gold=[10, 10, 70, 10, 10, 14, 14, 4, 6])
    assert _col(clean_game(rows), "gold") == [10, 10, 10, 10, 10, 14, 14, 4, 6]


def test_leading_gap_stays_none():
    rows = _rows(gold=[None, None, 5, 5])
    cleaned = clean_game(rows)
    assert _col(cleaned, "gold") == [None, None, 5, 5]
    assert cleaned[0]["filled"] == []


def test_rows_are_sorted_and_input_untouched():
    rows = _rows(hp=[100, 90, 80])
    shuffled = [rows[2], rows[0], rows[1]]
    cleaned = clean_game(shuffled)
    assert _col(cleaned, "frame") == [0, 1, 2]
    assert "filled" not in rows[0]


def test_quality_report_counts_reads_and_corrections():
    rows = _rows(stage=[2, 2, 2, 8, 2, 2, 2], round_num=[1] * 7, hp=[100] * 6 + [None],
                 gold=[5] * 7)
    rows[0]["shop_visible"] = False
    rows[0]["gold"] = None
    report = quality_report(rows, clean_game(rows))
    assert report["frames"] == 7 and report["bottom_bar_visible"] == 6
    assert report["fields"]["stage"]["misreads_corrected"] == 1
    assert report["fields"]["hp"]["read_rate"] == round(6 / 7, 3)
    assert report["fields"]["gold"]["read_rate"] == 1.0


def test_augments_carry_forward_when_bar_hidden():
    # Augments are decided per game, per bar position: once picked they stay,
    # whatever single frames show (bar hidden, an icon missed).
    rows = _rows(hp=[100] * 7)
    seen = [[], ["a"], [], ["a", "b"], ["a"], ["a", "b"], []]
    for row, augs in zip(rows, seen):
        row["augments"], row["augment_tiers"] = augs, ["gold"] * len(augs)
    cleaned = clean_game(rows)
    assert _col(cleaned, "augments") == [[], ["a"], ["a"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b"]]
    assert cleaned[5]["augment_tiers"] == ["gold", "gold"]
    assert "augments" in cleaned[2]["filled"] and "augments" not in cleaned[3]["filled"]


def test_no_augment_before_its_pick_round():
    # Hamza's checks (2026-11-25): a 2nd augment shown at 3-1, a 3rd at 1-4. Picks are at
    # 2-1, 3-2 and 4-2; early misreads must not move a slot's start before its round.
    stages = [(1, 4), (2, 1), (2, 5), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2)]
    rows = _rows(hp=[100] * len(stages), stage=[s for s, _ in stages], round_num=[r for _, r in stages])
    seen = [["a", "b", "c"], ["a"], ["a", "b"], ["a", "b"], ["a", "b"], ["a", "b", "c"],
            ["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
    for row, augs in zip(rows, seen):
        row["augments"], row["augment_tiers"] = augs, ["gold"] * len(augs)
    held = [len(r["augments"]) for r in clean_game(rows)]
    assert held == [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]


def test_augment_name_is_the_most_common_reading_and_labels_win():
    rows = _rows(hp=[100] * 5)
    seen = [["a"], ["a"], ["x"], ["a"], ["a"]]          # one misread
    for row, augs in zip(rows, seen):
        row["augments"], row["augment_tiers"] = augs, ["gold"] * len(augs)
    assert _col(clean_game(rows), "augments") == [["a"]] * 5
    labelled = clean_game(rows, {"augments": ["Cursed Crown"], "tiers": ["gold"]})
    assert _col(labelled, "augments") == [["Cursed Crown"]] * 5


def test_one_phantom_icon_is_not_a_pick():
    rows = _rows(hp=[100] * 4)
    seen = [["a"], ["a", "q"], ["a"], ["a"]]
    for row, augs in zip(rows, seen):
        row["augments"], row["augment_tiers"] = augs, ["gold"] * len(augs)
    assert _col(clean_game(rows), "augments") == [["a"]] * 4


def test_home_board_uses_augment_side_then_arena_colour():
    rows = _rows(hp=[100] * 5)
    own, other = [10.0] * 12, [200.0] * 12
    setup = [
        (["a"], 0, own),     # own bar on the left
        ([], 2, own),        # other player's bar on the right: away, whatever the colours say
        (["a"], 0, other),   # own bar showing on a dimmed arena: still home
        ([], 0, own),        # no bar: arena colour decides -> home
        ([], 0, other),      # no bar: arena colour decides -> away
    ]
    for row, (augs, away, sig) in zip(rows, setup):
        row.update(augments=augs, augment_tiers=["gold"] * len(augs), away_augments=away,
                   arena_sig=sig)
    assert _col(clean_game(rows), "home_board") == [True, False, True, True, False]


def test_augments_seen_while_away_are_not_taken():
    rows = _rows(hp=[100] * 3)
    rows = _rows(hp=[100] * 4)
    for row, (augs, away) in zip(rows, [(["a"], 0), (["x", "y", "z"], 3), ([], 0), (["a"], 0)]):
        row.update(augments=augs, augment_tiers=["gold"] * len(augs), away_augments=away)
    assert _col(clean_game(rows), "augments") == [["a"], ["a"], ["a"], ["a"]]


def test_planning_ends_at_first_timer_refill_in_a_round():
    rows = _rows(stage=[3] * 6 + [3] * 3, round_num=[1] * 6 + [2] * 3)
    fills = [0.9, 0.5, 0.1, 0.95, 0.6, 0.2,    # round 3-1: drains, refills (combat), drains
             0.8, 0.4, 0.1]                    # round 3-2: planning again
    for row, fill in zip(rows, fills):
        row.update(timer_fill=fill, planning=True)
    rows[7]["planning"] = False               # health bars on the far side: not planning
    assert _col(clean_game(rows), "planning") == [True, True, True, False, False, False,
                                                  True, False, True]


def test_hidden_timer_at_round_start_is_not_a_refill():
    # 3-2: the augment choice hides the timer (reads 0.0); it then shows at 0.85.
    # That is not combat starting; the real refill comes after the drain.
    rows = _rows(stage=[3] * 8, round_num=[2] * 8)
    fills = [0.0, 0.85, 0.0, 0.77, 0.5, 0.1, 0.01, 0.52]
    for row, fill in zip(rows, fills):
        row.update(timer_fill=fill, planning=True)
    assert _col(clean_game(rows), "planning") == [False, True, False, True, True, True, False, False]


# ── own_span: his own game only ───────────────────────────────────────────────

def _hud(n, stage, rnd, hp, level):
    return [{"frame": 0, "stage": stage, "round_num": rnd, "hp": hp, "level": level}] * n


def _framed(rows):
    return [dict(r, frame=i) for i, r in enumerate(rows)]


def test_own_span_cuts_another_players_hud_after_elimination():
    from src.state.sequence import own_span
    # HP 7 at 4-4, the death screen (nothing read), then another player at HP 38, level 8
    rows = _framed(_hud(30, 4, 4, 7, 7) + [{"frame": 0}] * 10 + _hud(40, 5, 5, 38, 8))
    start, end, reason = own_span(rows)
    assert (start, end) == (0, 30) and reason.startswith("HP 7 -> 38")


def test_own_span_ignores_a_short_misread():
    from src.state.sequence import own_span
    # level misread 5 on three frames running at level 8 (--ppQShtXow/game_08)
    rows = _framed(_hud(30, 5, 7, 12, 8) + _hud(3, 5, 7, 12, 5) + _hud(40, 6, 1, 12, 8))
    assert own_span(rows) == (0, len(rows), None)
    # a round or two back is OCR (6-3 settled among 6-5s), not another game
    rows = _framed(_hud(20, 6, 5, 19, 9) + _hud(4, 6, 3, 19, 9) + _hud(20, 6, 5, 19, 9))
    assert own_span(rows) == (0, len(rows), None)


def test_own_span_drops_the_previous_games_tail():
    from src.state.sequence import own_span
    rows = _framed(_hud(10, 6, 5, 4, 9) + _hud(40, 1, 1, 100, 1) + _hud(40, 2, 1, 100, 3))
    start, end, reason = own_span(rows)
    assert (start, end) == (10, len(rows)) and reason is None


def test_own_span_keeps_a_normal_game():
    from src.state.sequence import own_span
    rows = _framed(_hud(20, 1, 1, 100, 1) + _hud(20, 2, 1, 90, 4) + _hud(20, 3, 2, 70, 6)
                   + _hud(20, 4, 2, 72, 7) + _hud(20, 5, 1, 40, 8))      # a small heal is fine
    assert own_span(rows) == (0, len(rows), None)
