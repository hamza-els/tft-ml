"""
Tests for game data lookup and the state encoder, using a tiny made-up set.

Usage:
    python -m pytest tests/test_state/test_encoder.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from src.state.encoder import StateEncoder, round_index
from src.state.game_data import GameData, load_game_data

RAW = {
    "set": "TestSet",
    "champions": {
        "Ahri":        {"cost": 4, "traits": ["Fox", "Mage"]},
        "Lux":         {"cost": 5, "traits": ["Avatar"]},
        "Lux (Fox)":   {"cost": 5, "traits": ["Fox", "Avatar"]},
        "Shen":        {"cost": 2, "traits": ["Guard"]},
        "Teemo":       {"cost": 1, "traits": ["Fox", "Guard"]},
    },
    "traits": {
        "Avatar": {"breakpoints": [1]},
        "Fox":    {"breakpoints": [2, 4]},
        "Guard":  {"breakpoints": [2, 4, 6]},
        "Mage":   {"breakpoints": [2]},
    },
}


@pytest.fixture
def enc():
    return StateEncoder(GameData(RAW))


def _get(enc, vec, name):
    return float(vec[enc.feature_names.index(name)])


def _unit(name, star=None):
    return {"name": name, "star": star, "items": [], "region": "board"}


# ── GameData ──────────────────────────────────────────────────────────────────

def test_form_variants_collapse_to_base_name():
    game = GameData(RAW)
    assert game.champion_names == ["Ahri", "Lux", "Shen", "Teemo"]
    assert game.champions["Lux"].traits == ("Avatar",)


def test_trait_counts_ignore_duplicate_units():
    game = GameData(RAW)
    assert game.trait_counts(["Teemo", "Teemo", "Ahri"]) == {"Fox": 2, "Guard": 1, "Mage": 1}


def test_active_tier():
    game = GameData(RAW)
    assert [game.active_tier("Guard", n) for n in (1, 2, 3, 4, 6)] == [0, 1, 1, 2, 3]


def test_unknown_champion_costs_nothing():
    assert GameData(RAW).cost("Blast Potion") == 0


# ── encoder ───────────────────────────────────────────────────────────────────

def test_round_index():
    assert round_index(1, 1) == 0
    assert round_index(1, 4) == 3
    assert round_index(2, 1) == 4
    assert round_index(3, 2) == 12
    assert round_index(None, 2) is None


def test_vector_size_matches_names_and_blocks(enc):
    vec = enc.encode({})
    assert vec.shape == (enc.size,) == (len(enc.feature_names),)
    assert sum(length for _, _, length in enc.describe()) == enc.size
    assert not vec.any(), "an empty row must encode to all zeros"


def test_economy_features(enc):
    vec = enc.encode({"gold": 34, "level": 6, "hp": 80, "xp_current": 9, "xp_needed": 36,
                      "streak": -3, "stage": 3, "round_num": 2})
    assert _get(enc, vec, "gold") == pytest.approx(0.34)
    assert _get(enc, vec, "interest") == pytest.approx(3 / 5)
    assert _get(enc, vec, "gold_to_next_interest") == pytest.approx(0.6)
    assert _get(enc, vec, "xp_fraction") == pytest.approx(0.25)
    assert _get(enc, vec, "loss_streak") == pytest.approx(0.3)
    assert _get(enc, vec, "win_streak") == 0.0
    assert _get(enc, vec, "round_index") == pytest.approx(12 / 46)


def test_interest_caps_at_fifty_gold(enc):
    vec = enc.encode({"gold": 73})
    assert _get(enc, vec, "interest") == 1.0
    assert _get(enc, vec, "gold_to_next_interest") == 0.0


def test_shop_block_carries_cost_and_traits(enc):
    vec = enc.encode({"shop_slots": ["Teemo", None, "Ahri", "Teemo", None], "shop_visible": True})
    assert _get(enc, vec, "shop_slot_0_cost") == pytest.approx(1 / 5)
    assert _get(enc, vec, "shop_slot_1_cost") == 0.0
    assert _get(enc, vec, "shop_slot_2_cost") == pytest.approx(4 / 5)
    assert _get(enc, vec, "shop_has_Teemo") == pytest.approx(2 / 5)
    assert _get(enc, vec, "shop_cost_1") == pytest.approx(2 / 5)
    assert _get(enc, vec, "shop_filled") == pytest.approx(3 / 5)
    assert _get(enc, vec, "shop_trait_Fox") == pytest.approx(2 / 4)   # Teemo + Ahri of max 4
    assert _get(enc, vec, "bottom_bar_visible") == 1.0


def test_shop_matches_units_already_owned(enc):
    row = {"shop_slots": ["Teemo", "Shen", "Ahri", None, None],
           "board": [_unit("Teemo")], "bench": [_unit("Shen")]}
    vec = enc.encode(row, has_board=True)
    assert _get(enc, vec, "shop_matches_owned") == pytest.approx(2 / 5)
    # Teemo (Fox, Guard) is fielded: Teemo, Shen (Guard) and Ahri (Fox) all share a trait
    assert _get(enc, vec, "shop_matches_board_trait") == pytest.approx(3 / 5)


def test_board_block(enc):
    row = {"level": 5, "board": [_unit("Teemo", 2), _unit("Shen"), _unit("Ahri")],
           "bench": [_unit("Teemo")]}
    vec = enc.encode(row, has_board=True)
    assert _get(enc, vec, "board_units") == pytest.approx(3 / 12)
    assert _get(enc, vec, "board_slots_free") == pytest.approx(2 / 10)
    assert _get(enc, vec, "board_value") == pytest.approx((1 * 3 + 2 + 4) / 150)
    assert _get(enc, vec, "board_tier_Fox") == pytest.approx(1 / 2)    # 2 Fox: first of 2 tiers
    assert _get(enc, vec, "board_tier_Guard") == pytest.approx(1 / 3)
    assert _get(enc, vec, "board_tier_Mage") == 0.0                    # 1 Mage, needs 2
    assert _get(enc, vec, "board_active_traits") == pytest.approx(2 / 10)
    assert _get(enc, vec, "bench_has_Teemo") == pytest.approx(1 / 3)
    assert _get(enc, vec, "has_board") == 1.0


def test_lobby_block(enc):
    vec = enc.encode({"hp": 50, "other_hp": [80, 60, 40, 0]})
    assert _get(enc, vec, "other_hp_0") == pytest.approx(0.8)
    assert _get(enc, vec, "players_alive") == pytest.approx(3 / 7)
    assert _get(enc, vec, "hp_rank") == pytest.approx(2 / 7)           # two players above 50
    assert _get(enc, vec, "has_lobby") == 1.0


def test_values_stay_in_range(enc):
    row = {"gold": 500, "hp": 100, "level": 10, "streak": 30, "stage": 9, "round_num": 9,
           "xp_current": 5, "xp_needed": 4, "other_hp": [100] * 9,
           "shop_slots": ["Lux"] * 5, "board": [_unit("Lux", 3)] * 14}
    vec = enc.encode(row, has_board=True)
    assert vec.min() >= -1.0 and vec.max() <= 1.0


def test_encode_game_uses_time_since_first_frame(enc):
    rows = [{"time_sec": 1000.0}, {"time_sec": 1600.0}]
    X = enc.encode_game(rows)
    assert X.shape == (2, enc.size)
    i = enc.feature_names.index("time_in_game")
    assert X[0, i] == 0.0 and X[1, i] == pytest.approx(600 / 2400)


def test_real_set_data_loads():
    game = load_game_data()
    assert game.cost("Morgana") == 4
    assert set(game.champions["Morgana"].traits) == {"Coven", "Invoker"}
    assert "Lux" in game.champions and not any("(" in n for n in game.champion_names)


def test_augment_block(enc):
    vec = enc.encode({"augments": ["aug_001", None], "augment_tiers": ["gold", "prismatic"]})
    assert _get(enc, vec, "augment_count") == pytest.approx(2 / 3)
    assert _get(enc, vec, "augments_gold") == pytest.approx(1 / 3)
    assert _get(enc, vec, "augments_prismatic") == pytest.approx(1 / 3)
    assert _get(enc, vec, "augments_silver") == 0.0


def test_item_block():
    items = {"B.F. Sword": {"recipe": []}, "Deathblade": {"recipe": ["B.F. Sword", "B.F. Sword"]},
             "Reforger": {"recipe": []}}
    enc = StateEncoder(GameData(RAW, items))
    vec = enc.encode({"bench_items": ["B.F. Sword", "B.F. Sword", "Deathblade", "Reforger", None]})
    assert _get(enc, vec, "items_held") == pytest.approx(5 / 10)
    assert _get(enc, vec, "item_components") == pytest.approx(2 / 10)
    assert _get(enc, vec, "items_completed") == pytest.approx(1 / 10)
    assert _get(enc, vec, "items_other") == pytest.approx(2 / 10)    # consumable + unknown icon


def test_extra_units_count_on_board_but_not_in_shop():
    enc = StateEncoder(GameData(RAW, extra_units=["Stonebark Tree", "Ahri"]))
    assert enc.game.extra_units == ["Stonebark Tree"]           # Ahri is already a champion
    assert "board_has_Stonebark Tree" in enc.feature_names
    assert "shop_has_Stonebark Tree" not in enc.feature_names
    vec = enc.encode({"board": [_unit("Stonebark Tree"), _unit("Teemo")]}, has_board=True)
    assert _get(enc, vec, "board_has_Stonebark Tree") == pytest.approx(1 / 3)
    assert _get(enc, vec, "board_units") == pytest.approx(2 / 12)
    assert enc.game.cost("Stonebark Tree") == 0
