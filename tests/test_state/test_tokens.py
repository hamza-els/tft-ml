"""
Per-unit token table (src/state/tokens.py).

Usage:
    python -m pytest tests/test_state/test_tokens.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from src.state.tokens import INT_FIELDS, LIMITS, MAX_ROWS, TokenEncoder, placement


@pytest.fixture(scope="module")
def enc():
    return TokenEncoder()


def row():
    return {
        "board": [{"name": "Taric", "star": 2, "items": ["Radiant Warmog's Armor", "Fae Emblem"],
                   "hex": [3, 1], "p": 0.9, "carried": True}],
        "bench": [{"name": "Ashe", "star": 1, "items": [], "slot": 4, "p": 0.8, "filled": True}],
        "shop_slots": ["Gnar", None, "Rakan", None, None],
        "bench_items": ["B.F. Sword", None],
    }


def get(enc, tok_int, tok_float, k, field):
    if field in INT_FIELDS:
        return int(tok_int[k, INT_FIELDS.index(field)])
    return float(tok_float[k, enc.float_fields.index(field)])


def test_rows_and_padding(enc):
    ti, tf, mask = enc.encode(row())
    assert ti.shape == (MAX_ROWS, len(INT_FIELDS)) and tf.shape[0] == MAX_ROWS
    # 1 board, 1 bench, 2 shop cards (empty slots skipped), 2 item column entries
    assert mask.sum() == 6
    assert [get(enc, ti, tf, k, "group") for k in range(6)] == [1, 2, 3, 3, 4, 4]
    assert get(enc, ti, tf, 6, "group") == 0 and not mask[6]


def test_board_unit(enc):
    ti, tf, _ = enc.encode(row())
    assert enc.units[get(enc, ti, tf, 0, "unit")] == "Taric"
    assert get(enc, ti, tf, 0, "star") == 2
    assert enc.items[get(enc, ti, tf, 0, "item0")] == "Warmog's Armor"
    assert get(enc, ti, tf, 0, "radiant0") == 1
    assert enc.items[get(enc, ti, tf, 0, "item1")] == "Fae Emblem" and get(enc, ti, tf, 0, "radiant1") == 0
    assert get(enc, ti, tf, 0, "item2") == 0                       # empty third slot = padding id
    assert (get(enc, ti, tf, 0, "hex_row"), get(enc, ti, tf, 0, "hex_col")) == (3, 1)
    assert get(enc, ti, tf, 0, "carried") == 1.0 and abs(get(enc, ti, tf, 0, "confidence") - 0.9) < 1e-6
    # the Fae Emblem adds Fae to Taric's own traits
    assert get(enc, ti, tf, 0, "trait_Fae") == 1.0
    for t in enc.game.champions["Taric"].traits:
        assert get(enc, ti, tf, 0, f"trait_{t}") == 1.0


def test_bench_shop_items(enc):
    ti, tf, _ = enc.encode(row())
    assert enc.units[get(enc, ti, tf, 1, "unit")] == "Ashe" and get(enc, ti, tf, 1, "slot") == 4
    assert get(enc, ti, tf, 1, "filled") == 1.0 and get(enc, ti, tf, 1, "hex_row") == -1
    assert enc.units[get(enc, ti, tf, 3, "unit")] == "Rakan" and get(enc, ti, tf, 3, "slot") == 2
    assert enc.items[get(enc, ti, tf, 4, "item")] == "B.F. Sword"
    assert enc.items[get(enc, ti, tf, 5, "item")] == "_unknown"     # an icon with no name


def test_unknown_names(enc):
    ti, tf, _ = enc.encode({"board": [{"name": "_unknown", "star": None, "items": ["uitem_999"], "hex": [0, 0], "p": 0.5}]})
    assert enc.units[get(enc, ti, tf, 0, "unit")] == "_unknown"
    assert enc.items[get(enc, ti, tf, 0, "item0")] == "_unknown"
    assert get(enc, ti, tf, 0, "star") == 0


def test_vocab_fixed_by_set_data(enc):
    again = TokenEncoder()
    assert again.units == enc.units and again.items == enc.items
    assert enc.units[0] == "<pad>" and enc.items[0] == "<pad>"
    assert not any(n.startswith("Radiant ") and n[8:] in enc.items for n in enc.items)


def test_board_limit(enc):
    many = [{"name": "Ashe", "star": 1, "items": [], "hex": [k // 7, k % 7], "p": k / 30} for k in range(20)]
    _, _, mask = enc.encode({"board": many})
    assert mask.sum() == LIMITS["board"]


def test_placement_label():
    rows = [{"other_hp": [100] * 7}, {"other_hp": [50, 0, 0, 20, 0, 0, 0]}]
    assert placement(rows) == 3
    assert placement([{}]) == -1
