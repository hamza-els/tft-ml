"""
Per-frame token table: one row per board unit, bench unit, shop card and
unequipped item, next to the flat state vector (src/state/encoder.py).

The flat vector only summarises the board (copies, cost mix, traits). The
table keeps each champion with its own star level, items and position, for a
set / transformer model (docs/gameplay_model.md). Decided with Hamza
(2026-11-19):

  - identity as integer IDs the model learns embeddings for (vocabularies
    fixed by the set data: GameData.unit_vocab / item_vocab, 0 = padding);
  - Radiant items as their base item's ID plus a radiant flag;
  - position and reading certainty per row;
  - each champion's cost and traits as fixed inputs, so a rare champion's
    traits are known without needing many examples to learn them.

Rows, in order, padded to MAX_ROWS:
    group 1 board  (up to 14, by hex)       champion, star, items (a set: sorted by ID), hex
    group 2 bench  (up to 9)                champion, star, items (slot left out: meaningless)
    group 3 shop   (up to 5, by slot)       champion, slot
    group 4 item   (up to 10)               item   (the unequipped item column; no slot: meaningless)
    group 5 augment (up to 3)               augment, tier (1 silver, 2 gold, 3 prismatic), slot = pick order

Augment tiers are variations of one augment (Hamza, 2026-11-19): "New Recruit+"
and "Celestial Blessing II" are their base augment's ID with a tier.

tok_int fields (int16, -1 = not applicable): see INT_FIELDS.
tok_float fields (float32): see FLOAT_FIELDS.
"""

import numpy as np

from src.state.game_data import GameData, load_game_data

GROUPS = {"board": 1, "bench": 2, "shop": 3, "item": 4, "augment": 5}
LIMITS = {"board": 14, "bench": 9, "shop": 5, "item": 10, "augment": 4}   # augments: 3 picks + 1 extra (seen in footage)
MAX_ROWS = sum(LIMITS.values())
INT_FIELDS = ["group", "unit", "star", "item0", "item1", "item2", "radiant0", "radiant1", "radiant2",
              "item", "item_radiant", "hex_row", "hex_col", "slot", "augment", "augment_tier"]
AUGMENT_TIERS = {"silver": 1, "gold": 2, "prismatic": 3}
CERTAINTY = ["confidence", "carried", "filled", "loose"]


class TokenEncoder:
    def __init__(self, game: GameData | None = None):
        self.game = game or load_game_data()
        self.units = self.game.unit_vocab()
        self.items = self.game.item_vocab()
        self.unit_id = {n: i for i, n in enumerate(self.units)}
        self.item_id = {n: i for i, n in enumerate(self.items)}
        self.augments = self.game.augment_vocab()
        self.augment_id = {n: i for i, n in enumerate(self.augments)}
        self.traits = self.game.trait_names
        self.float_fields = CERTAINTY + ["cost"] + [f"trait_{t}" for t in self.traits]
        self._f = {n: i for i, n in enumerate(self.float_fields)}
        self._i = {n: i for i, n in enumerate(INT_FIELDS)}

    # ── ids ───────────────────────────────────────────────────────────────────

    def unit(self, name: str | None) -> int:
        return self.unit_id.get(name, self.unit_id["_unknown"]) if name else self.unit_id["_unknown"]

    def item(self, name: str | None) -> tuple[int, int]:
        """(item ID, radiant); an icon read but not recognised (None) or an unnamed ID is _unknown."""
        base, radiant = self.game.item_base(name.split("~")[0] if name else None)
        return self.item_id.get(base, self.item_id["_unknown"]), int(radiant)

    def augment(self, name: str | None) -> int:
        """Base augment ID; unnamed template IDs (aug_XXX) and unread icons are _unknown."""
        return self.augment_id.get(self.game.augment_base(name), self.augment_id["_unknown"])

    # ── rows ──────────────────────────────────────────────────────────────────

    def _champion_row(self, group: str, name: str, unit: dict | None) -> tuple[list, list]:
        ints = [-1] * len(INT_FIELDS)
        floats = [0.0] * len(self.float_fields)
        ints[self._i["group"]] = GROUPS[group]
        ints[self._i["unit"]] = self.unit(name)
        held = []
        if unit is not None:
            ints[self._i["star"]] = int(unit.get("star") or 0)
            held = list(unit.get("items") or [])[:3]
            # Items held are a set (Hamza, 2026-11-19: which slot an item sits
            # in means nothing): sorted by item ID, each with its radiant flag,
            # so the order carries no information.
            pairs = sorted(self.item(i) for i in held)
            for k in range(3):
                iid, rad = pairs[k] if k < len(pairs) else (0, 0)
                ints[self._i[f"item{k}"]] = iid
                ints[self._i[f"radiant{k}"]] = rad
            if "hex" in unit:
                ints[self._i["hex_row"]], ints[self._i["hex_col"]] = int(unit["hex"][0]), int(unit["hex"][1])
            # The bench slot a unit sits in carries no meaning (Hamza, 2026-11-19);
            # only shop slots keep theirs (buying is per slot).
            floats[self._f["confidence"]] = float(unit.get("p", 1.0))
            for flag in ("carried", "filled", "loose"):
                floats[self._f[flag]] = float(bool(unit.get(flag)))
        else:
            ints[self._i["star"]] = 1                      # shop cards are 1-star
            floats[self._f["confidence"]] = 1.0
        floats[self._f["cost"]] = self.game.cost(name) / 5.0
        # The champion's own traits, plus any an emblem it holds adds.
        champ = self.game.champions.get(name)
        traits = set(champ.traits) if champ else set()
        traits |= {self.game.emblems[i] for i in held if i in self.game.emblems}
        for t in traits:
            if f"trait_{t}" in self._f:
                floats[self._f[f"trait_{t}"]] = 1.0
        return ints, floats

    def _item_row(self, name: str) -> tuple[list, list]:
        """An unequipped item. Its place in the item column means nothing (Hamza, 2026-11-19): no slot."""
        ints = [-1] * len(INT_FIELDS)
        floats = [0.0] * len(self.float_fields)
        ints[self._i["group"]] = GROUPS["item"]
        ints[self._i["item"]], ints[self._i["item_radiant"]] = self.item(name)
        floats[self._f["confidence"]] = 1.0
        return ints, floats

    def encode(self, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tok_int [MAX_ROWS, len(INT_FIELDS)], tok_float [MAX_ROWS, F], mask [MAX_ROWS]) for one cleaned row."""
        rows: list[tuple[list, list]] = []
        board = sorted(row.get("board") or [], key=lambda u: -u.get("p", 0))[:LIMITS["board"]]
        for u in sorted(board, key=lambda u: tuple(u.get("hex", (9, 9)))):
            rows.append(self._champion_row("board", u["name"], u))
        for u in sorted(row.get("bench") or [], key=lambda u: u.get("slot", 9))[:LIMITS["bench"]]:
            rows.append(self._champion_row("bench", u["name"], u))
        for slot, name in enumerate((row.get("shop_slots") or [])[:LIMITS["shop"]]):
            if name:
                ints, floats = self._champion_row("shop", name, None)
                ints[self._i["slot"]] = slot
                rows.append((ints, floats))
        for name in (row.get("bench_items") or [])[:LIMITS["item"]]:
            rows.append(self._item_row(name))
        tiers = row.get("augment_tiers") or []
        for k, name in enumerate((row.get("augments") or [])[:LIMITS["augment"]]):
            ints = [-1] * len(INT_FIELDS)
            floats = [0.0] * len(self.float_fields)
            ints[self._i["group"]] = GROUPS["augment"]
            ints[self._i["augment"]] = self.augment(name)
            ints[self._i["augment_tier"]] = AUGMENT_TIERS.get(tiers[k] if k < len(tiers) else None, 0)
            ints[self._i["slot"]] = k                   # pick order: 2-1, 3-2, 4-2
            floats[self._f["confidence"]] = 1.0
            rows.append((ints, floats))

        tok_int = np.full((MAX_ROWS, len(INT_FIELDS)), -1, np.int16)
        tok_int[:, self._i["group"]] = 0
        tok_float = np.zeros((MAX_ROWS, len(self.float_fields)), np.float32)
        mask = np.zeros(MAX_ROWS, bool)
        for k, (ints, floats) in enumerate(rows[:MAX_ROWS]):
            tok_int[k], tok_float[k], mask[k] = ints, floats, True
        return tok_int, tok_float, mask

    def encode_game(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not rows:
            return (np.zeros((0, MAX_ROWS, len(INT_FIELDS)), np.int16),
                    np.zeros((0, MAX_ROWS, len(self.float_fields)), np.float32),
                    np.zeros((0, MAX_ROWS), bool))
        parts = [self.encode(r) for r in rows]
        return tuple(np.stack(p) for p in zip(*parts))

    def vocab(self) -> dict:
        return {"units": self.units, "items": self.items, "augments": self.augments, "traits": self.traits,
                "int_fields": INT_FIELDS, "float_fields": self.float_fields,
                "groups": GROUPS, "limits": LIMITS}

    def describe(self) -> list[str]:
        return ([f"{MAX_ROWS} rows per frame: " + ", ".join(f"{g} {n}" for g, n in LIMITS.items()),
                 f"tok_int   {len(INT_FIELDS)}: " + ", ".join(INT_FIELDS),
                 f"tok_float {len(self.float_fields)}: " + ", ".join(CERTAINTY + ["cost"])
                 + f", trait_* x{len(self.traits)}",
                 f"vocab: {len(self.units)} units, {len(self.items)} items, {len(self.augments)} augments (0 = padding)"])


def placement(rows: list[dict]) -> int:
    """Final placement label: 1 + other players still alive at the last frame with lobby HP (-1 if unknown)."""
    for r in reversed(rows):
        others = r.get("other_hp") or []
        if others:
            return 1 + sum(1 for h in others if h and h > 0)
    return -1
