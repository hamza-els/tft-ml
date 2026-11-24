"""
Purchases (src/state/events.py), items over time (src/state/items.py) and the
2026-11-24 cleaning rules that use them (src/state/units.py).

Usage:
    python -m pytest tests/test_state/test_events.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.state.events import purchases
from src.state.game_data import load_game_data
from src.state.units import clean_units

GD = load_game_data()
SHOP = ["Akali", "Camille", "Karma", "Kobuko", "Leona"]       # 1-costs


def _shop_rows(shops, golds):
    return [{"frame": i, "shop_visible": True, "shop_slots": s, "gold": g} for i, (s, g) in enumerate(zip(shops, golds))]


def test_purchase_found_when_a_card_empties_and_gold_drops():
    bought = SHOP[:1] + [None] + SHOP[2:]
    rows = _shop_rows([SHOP, bought, bought], [10, 9, 9])
    p = purchases(rows, GD)
    assert [(x["name"], x["slot"], x["confirmed"]) for x in p] == [("Camille", 1, True)]


def test_reroll_sell_panel_and_flicker_are_not_purchases():
    other = ["Vi", "Lux", "Sett", "Ahri", "Taric"]
    assert purchases(_shop_rows([SHOP, other, other], [10, 8, 8]), GD) == []                 # reroll
    assert purchases(_shop_rows([SHOP, [None] * 5, SHOP], [10, 10, 10]), GD) == []            # sell panel
    flick = SHOP[:1] + [None] + SHOP[2:]
    assert purchases(_shop_rows([SHOP, flick, SHOP], [10, 10, 10]), GD) == []                  # back next frame
    assert purchases(_shop_rows([SHOP, flick, flick], [10, 10, 10]), GD) == []                 # no gold spent


def _u(at, name, p=0.95, star=1, items=(), **kw):
    return {"at": at, "name": name, "p": p, "star": star, "items": list(items), **kw}


def test_bought_unit_placed_on_the_leftmost_empty_bench_slot():
    rows = [{"frame": i * 2, "units_raw": [_u("s0", "Vi")]} for i in range(4)]
    bought = [[], ["Camille"], [], []]
    out, stats = clean_units(rows, [None] * 4, GD, [(2, 2)] * 4, bought=bought)
    assert [u["name"] for u in out[1]["bench"]] == ["Vi", "Camille"] and out[1]["bench"][1]["slot"] == 1
    assert stats["bought_filled"] >= 1


def test_bought_unit_misread_on_the_bench_takes_its_name():
    rows = [{"frame": 0, "units_raw": [_u("s0", "Vi")]},
            {"frame": 2, "units_raw": [_u("s0", "Vi"), _u("s1", "Lux", p=0.6, alt=[["Camille", 0.3]])]}]
    out, stats = clean_units(rows, [None] * 2, GD, [(2, 2)] * 2, bought=[[], ["Camille"]])
    assert [u["name"] for u in out[1]["bench"]] == ["Vi", "Camille"] and stats["bought_renamed"] == 1


def test_standing_loose_unit_kept_without_the_counter_but_not_while_dragging():
    rows = [{"frame": i * 2, "units_raw": [_u("h00", "Vi"), _u("h36", "Sett", loose=True)]} for i in range(3)]
    out, _ = clean_units(rows, [None] * 3, GD, [(2, 2)] * 3)
    assert len(out[1]["board"]) == 2
    for r in rows:
        r.update(shop_visible=True, shop_slots=[None] * 5)          # the sell panel: a unit is held
    out, _ = clean_units(rows, [None] * 3, GD, [(2, 2)] * 3)
    assert [u["name"] for u in out[1]["board"]] == ["Vi"]


def test_stretch_carries_across_combat_when_the_bench_is_unchanged(monkeypatch):
    # off by default (it made held-out crops worse); the rule itself still works when switched on
    import src.state.units as U
    monkeypatch.setattr(U, "ACROSS_COMBAT", True)
    vi, sett = _u("h00", "Vi"), _u("h01", "Sett", p=0.92)
    boards = [[vi, sett]] * 3 + [[vi]] * 3          # Sett's bar hidden after the fight, round 2-3
    rows = [{"frame": i * 2, "units_raw": b + [_u("s0", "Lux")]} for i, b in enumerate(boards)]
    out, _ = clean_units(rows, [None] * 6, GD, [(2, 2)] * 3 + [(2, 3)] * 3, gold=[10] * 3 + [15] * 3)
    assert [len(o["board"]) for o in out] == [2] * 6


def test_item_kept_when_nothing_came_back_to_the_column_and_removal_seen_when_it_did():
    ie = ["Infinity Edge", "Bloodthirster"]
    rows = [{"frame": i * 2, "bench_items": [], "units_raw": [_u("h00", "Vi", items=ie if i < 3 else ie[:1])]}
            for i in range(5)]
    out, stats = clean_units(rows, [None] * 5, GD, [(2, 2)] * 5)
    assert out[3]["board"][0]["items"] == ie and out[3]["board"][0].get("items_kept")
    for r in rows[3:]:
        r["bench_items"] = ["Bloodthirster"]                          # removed: back in the column
    out, stats = clean_units(rows, [None] * 5, GD, [(2, 2)] * 5)
    assert out[3]["board"][0]["items"] == ie[:1]
    assert any(e["kind"] == "removed" for o in out for e in o.get("item_events", []))


def test_two_components_becoming_an_item_is_a_combine():
    rows = [{"frame": i * 2, "bench_items": [], "units_raw": [_u("h00", "Vi", items=["B.F. Sword"] if i < 2 else ["Infinity Edge"])]}
            for i in range(4)]
    out, stats = clean_units(rows, [None] * 4, GD, [(2, 2)] * 4)
    assert stats["item_combine"] == 1 and out[2]["board"][0]["items"] == ["Infinity Edge"]
