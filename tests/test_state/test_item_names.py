"""Item-name folding: bundles map to their item, loot rewards are no items, extras are items."""

from src.state.game_data import fold_item_bundles, load_game_data


def test_fold_item_bundles():
    items = {"Reforger": {}, "Tactician's Cape": {}, "Lesser Champion Duplicator": {},
             "2 Reforgers": {}, "2 Tacticians' Capes": {}, "6 Lesser Champion Duplicators": {},
             "10 gold": {}, "2-star 3-cost champion": {}, "4 components": {}}
    kept, aliases = fold_item_bundles(items)
    assert set(kept) == {"Reforger", "Tactician's Cape", "Lesser Champion Duplicator"}
    assert aliases == {"2 Reforgers": "Reforger", "2 Tacticians' Capes": "Tactician's Cape",
                       "6 Lesser Champion Duplicators": "Lesser Champion Duplicator",
                       "10 gold": None, "2-star 3-cost champion": None, "4 components": None}


def test_game_data_items():
    g = load_game_data()
    assert not [n for n in g.items if n[:1].isdigit()]
    assert not [n for n in g.item_vocab() if n[:1].isdigit()]
    assert g.canonical_item("2 Reforgers") == "Reforger"
    assert g.canonical_item("10 gold") is None
    assert g.canonical_item("Reforger") == "Reforger"
    assert "Alpha Mark" in g.items and "Alpha Mark" in g.item_vocab()
