"""
Tests for action derivation and the Macro Engine's input/output shapes.

Usage:
    python -m pytest tests/test_models -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.models.macro_engine.actions import POLICY_ACTIONS, ActionType, derive_action, derive_actions
from src.models.macro_engine.model import MacroEngine
from src.state.game_data import GameData

GAME = GameData({
    "set": "T",
    "champions": {"Ahri": {"cost": 4, "traits": ["Fox"]}, "Shen": {"cost": 2, "traits": ["Guard"]},
                  "Teemo": {"cost": 1, "traits": ["Fox"]}},
    "traits": {"Fox": {"breakpoints": [2]}, "Guard": {"breakpoints": [2]}},
})
SHOP = ["Ahri", "Shen", "Teemo", "Shen", "Teemo"]


def _row(**kw) -> dict:
    row = {"shop_visible": True, "stage": 3, "round_num": 2, "gold": 30, "level": 6,
           "xp_current": 4, "shop_slots": list(SHOP)}
    row.update(kw)
    return row


def test_pass():
    assert derive_action(_row(), _row(), GAME).type is ActionType.PASS


def test_buy_names_the_slot_and_champion():
    after = _row(gold=26, shop_slots=[None] + SHOP[1:])
    action = derive_action(_row(), after, GAME)
    assert (action.type, action.arg, action.champion) == (ActionType.BUY, 0, "Ahri")
    assert POLICY_ACTIONS[action.policy_index] == (ActionType.BUY, 0)


def test_buy_with_wrong_gold_change_is_unknown():
    after = _row(gold=29, shop_slots=[None] + SHOP[1:])
    assert derive_action(_row(), after, GAME).type is ActionType.UNKNOWN


def test_reroll():
    after = _row(gold=28, shop_slots=["Teemo", "Ahri", "Shen", "Shen", "Ahri"])
    assert derive_action(_row(), after, GAME).type is ActionType.REROLL


def test_buy_xp_by_xp_gain_or_level_up():
    assert derive_action(_row(), _row(gold=26, xp_current=8), GAME).type is ActionType.BUY_XP
    assert derive_action(_row(), _row(gold=26, level=7, xp_current=0), GAME).type is ActionType.BUY_XP


def test_sell_is_gold_rising_mid_round():
    assert derive_action(_row(), _row(gold=33), GAME).type is ActionType.SELL


def test_unknown_across_rounds_or_with_bar_hidden():
    assert derive_action(_row(), _row(round_num=3, gold=40), GAME).type is ActionType.UNKNOWN
    assert derive_action(_row(), _row(shop_visible=False), GAME).type is ActionType.UNKNOWN
    assert derive_action(_row(), _row(gold=None), GAME).type is ActionType.UNKNOWN


def test_two_actions_in_one_step_are_unknown():
    # bought Ahri (4) and rerolled (2) between two frames
    after = _row(gold=24, shop_slots=["Teemo", "Teemo", "Shen", "Ahri", "Shen"])
    assert derive_action(_row(), after, GAME).type is ActionType.UNKNOWN


def test_derive_actions_length():
    assert len(derive_actions([_row(), _row(), _row(gold=28)], GAME)) == 2


def test_model_output_shapes_and_causality():
    model = MacroEngine(state_dim=20, hidden_dim=32, num_heads=2, num_layers=1).eval()
    x = torch.randn(2, 7, 20)
    out = model(x)
    assert out["policy"].shape == (2, 7, len(POLICY_ACTIONS))
    assert out["placement"].shape == (2, 7, 8)
    assert out["hp_next"].shape == (2, 7)

    # Changing a later frame must not change earlier outputs.
    y = x.clone()
    y[:, 5:] += 1.0
    with torch.no_grad():
        a, b = model(x)["policy"], model(y)["policy"]
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5)
    assert not torch.allclose(a[:, 5:], b[:, 5:])
