"""
Turn one frame's state (a cleaned row from scripts/clean_states.py) into a
fixed-length numeric vector for the Macro Engine.

The vector is the concatenation of named blocks (see StateEncoder.blocks):

  scalars   what the HUD says: stage, gold, HP, level, XP, streak, plus
            derived economy numbers (interest, gold to next interest step)
  lobby     the other players' HP
  shop      what is on offer: which champions, their costs and traits, and
            how many offers match a unit the player already owns
  board     what is fielded: which units (shop champions plus trait/augment
            units such as the Stonebark Tree), cost mix, trait counts
            (board_trait_<trait>: units with the trait, emblems included,
            / its top breakpoint), active trait tiers (board_tier_<trait>:
            breakpoints reached / number of breakpoints), free team slots
            (team size incl. trait/item bonuses, minus slots used)
  bench     what is held in reserve (same unit list)
  augments  how many augments are picked and of which tier
  items     how many unequipped items are held, split into components and
            finished items
  masks     1/0 flags for whether each source was available in this frame,
            so the model can tell "zero" from "unknown"

Champion knowledge (cost, traits, breakpoints) comes from GameData, so a
champion is never just an ID: buying a unit changes cost and trait features
the model can generalise over.

Everything is scaled to roughly 0-1. Board and bench come from the cleaned
rows (scripts/read_units.py, then clean_states.py); a game without unit
readings has those blocks all zeros and the has_board mask at 0. Hex
positions and items on units are in the rows but not encoded yet.
"""

import numpy as np

from src.state.game_data import MAX_COST, GameData, load_game_data

MAX_LEVEL = 10
MAX_GOLD = 100
MAX_STREAK = 10
MAX_BOARD = 12
MAX_BENCH = 9
SHOP_SLOTS = 5
OTHER_PLAYERS = 7
GAME_SECONDS = 2400      # ~40 min, the long end of a game
STAGE_ONE_ROUNDS = 4     # stage 1 has 4 rounds, later stages have 7
ROUNDS_PER_STAGE = 7
MAX_ROUND_INDEX = STAGE_ONE_ROUNDS + 6 * ROUNDS_PER_STAGE   # through stage 7

SCALAR_NAMES = [
    "stage", "round_in_stage", "round_index", "time_in_game",
    "gold", "interest", "gold_to_next_interest",
    "hp", "level", "xp_fraction", "xp_to_level",
    "streak", "win_streak", "loss_streak",
]
LOBBY_NAMES = (
    [f"other_hp_{i}" for i in range(OTHER_PLAYERS)]
    + ["players_alive", "other_hp_mean", "hp_rank"]
)
MAX_AUGMENTS = 3
AUGMENT_TIERS = ("silver", "gold", "prismatic")
AUGMENT_NAMES = ["augment_count"] + [f"augments_{t}" for t in AUGMENT_TIERS]
MAX_ITEMS = 10
ITEM_NAMES = ["items_held", "item_components", "items_completed", "items_other"]
MASK_NAMES = ["bottom_bar_visible", "has_lobby", "has_board", "on_home_board", "planning"]


def round_index(stage: float | None, round_num: float | None) -> int | None:
    """Rounds played since the start of the game (1-1 = 0, 2-1 = 4, 3-1 = 11, ...)."""
    if stage is None or round_num is None:
        return None
    if stage <= 1:
        return int(round_num) - 1
    return STAGE_ONE_ROUNDS + (int(stage) - 2) * ROUNDS_PER_STAGE + int(round_num) - 1


def _clip(value: float, top: float) -> float:
    return float(min(max(value, 0.0), top) / top)


class StateEncoder:
    def __init__(self, game: GameData | None = None):
        self.game = game or load_game_data()
        n_champ = len(self.game.champion_names)
        n_trait = len(self.game.trait_names)

        champs, traits = self.game.champion_names, self.game.trait_names
        units = self.game.unit_names
        costs = [f"cost_{c}" for c in range(1, MAX_COST + 1)]
        # (block name, feature names) in vector order
        self.blocks: list[tuple[str, list[str]]] = [
            ("scalars", SCALAR_NAMES),
            ("lobby", LOBBY_NAMES),
            ("shop_slot_cost", [f"shop_slot_{i}_cost" for i in range(SHOP_SLOTS)]),
            ("shop_champions", [f"shop_has_{c}" for c in champs]),
            ("shop_cost_mix", [f"shop_{c}" for c in costs]),
            ("shop_traits", [f"shop_trait_{t}" for t in traits]),
            ("shop_summary", ["shop_filled", "shop_total_cost", "shop_matches_owned",
                              "shop_matches_board_trait"]),
            ("board_unit_ids", [f"board_has_{u}" for u in units]),
            ("board_cost_mix", [f"board_{c}" for c in costs]),
            ("board_traits", [f"board_trait_{t}" for t in traits]),
            ("board_trait_tiers", [f"board_tier_{t}" for t in traits]),
            ("board_summary", ["board_units", "board_slots_free", "board_value",
                               "board_active_traits"]),
            ("bench_unit_ids", [f"bench_has_{u}" for u in units]),
            ("bench_summary", ["bench_units", "bench_value"]),
            ("augments", AUGMENT_NAMES),
            ("items", ITEM_NAMES),
            ("masks", MASK_NAMES),
        ]
        self.feature_names: list[str] = [n for _, names in self.blocks for n in names]
        self.size = len(self.feature_names)
        assert self.size == len(set(self.feature_names))
        self._n_champ, self._n_trait = n_champ, n_trait

    # ── block builders ────────────────────────────────────────────────────────

    def _scalars(self, row: dict, game_start_sec: float) -> list[float]:
        stage, rnd = row.get("stage"), row.get("round_num")
        idx = round_index(stage, rnd)
        gold = row.get("gold") or 0.0
        level = row.get("level") or 0.0
        xp_cur, xp_need = row.get("xp_current"), row.get("xp_needed")
        has_xp = xp_cur is not None and xp_need
        streak = row.get("streak") or 0.0
        return [
            _clip(stage or 0, 7),
            _clip(rnd or 0, ROUNDS_PER_STAGE),
            _clip(idx or 0, MAX_ROUND_INDEX),
            _clip(row.get("time_sec", game_start_sec) - game_start_sec, GAME_SECONDS),
            _clip(gold, MAX_GOLD),
            _clip(min(gold // 10, 5), 5),                 # interest earned at round end
            _clip(10 - gold % 10 if row.get("gold") is not None and gold < 50 else 0, 10),
            _clip(row.get("hp") or 0, 100),
            _clip(level, MAX_LEVEL),
            _clip(xp_cur / xp_need, 1) if has_xp else 0.0,
            _clip(xp_need - xp_cur, 100) if has_xp else 0.0,
            float(np.clip(streak / MAX_STREAK, -1, 1)),
            _clip(streak, MAX_STREAK),
            _clip(-streak, MAX_STREAK),
        ]

    def _lobby(self, row: dict) -> list[float]:
        others = sorted(row.get("other_hp") or [], reverse=True)[:OTHER_PLAYERS]
        padded = others + [0.0] * (OTHER_PLAYERS - len(others))
        alive = [h for h in others if h > 0]
        hp = row.get("hp") or 0.0
        return (
            [_clip(h, 100) for h in padded]
            + [_clip(len(alive), OTHER_PLAYERS),
               _clip(float(np.mean(alive)) if alive else 0, 100),
               # 0 = highest HP in the lobby, 1 = lowest
               _clip(sum(1 for h in alive if h > hp), OTHER_PLAYERS) if others else 0.0]
        )

    def _champion_counts(self, names: list[str], top: int, units: bool = False) -> np.ndarray:
        """Copies of each name / top. Shop uses the champion list; board and bench the unit list."""
        index = self.game.unit_index if units else self.game.champion_index
        vec = np.zeros(len(index), dtype=np.float32)
        for name in names:
            i = index.get(self.game.canonical_unit(name))
            if i is not None:
                vec[i] += 1.0 / top
        return np.minimum(vec, 1.0)

    def _cost_mix(self, names: list[str], top: int) -> list[float]:
        mix = [0.0] * MAX_COST
        for name in names:
            cost = self.game.cost(name)
            if cost:
                mix[cost - 1] += 1.0 / top
        return [min(m, 1.0) for m in mix]

    def _trait_vector(self, names: list[str], units: list[dict] | None = None) -> np.ndarray:
        """Units per trait (emblems included when units are given) / the trait's top breakpoint."""
        vec = np.zeros(self._n_trait, dtype=np.float32)
        for trait, count in self.game.trait_counts(names, units).items():
            top = max(self.game.breakpoints[trait]) if trait in self.game.breakpoints else 1
            if trait in self.game.trait_index:
                vec[self.game.trait_index[trait]] = min(count / top, 1.0)
        return vec

    def _trait_tiers(self, names: list[str], units: list[dict] | None = None) -> tuple[np.ndarray, int]:
        """Breakpoints reached per trait / number of breakpoints, and how many traits are active."""
        vec = np.zeros(self._n_trait, dtype=np.float32)
        active = 0
        for trait, count in self.game.trait_counts(names, units).items():
            if trait not in self.game.trait_index:
                continue
            tier = self.game.active_tier(trait, count)
            vec[self.game.trait_index[trait]] = tier / len(self.game.breakpoints[trait])
            active += tier > 0
        return vec, active

    def _items(self, items: list) -> list[float]:
        kinds = [self.game.item_kind(name) for name in items]
        return [
            _clip(len(items), MAX_ITEMS),
            _clip(kinds.count("component"), MAX_ITEMS),
            _clip(kinds.count("completed"), MAX_ITEMS),
            _clip(len(items) - kinds.count("component") - kinds.count("completed"), MAX_ITEMS),
        ]

    def _value(self, units: list[dict]) -> float:
        """Gold tied up in units: a 2-star is 3 copies, a 3-star is 9."""
        return sum(self.game.cost(u["name"]) * 3 ** ((u.get("star") or 1) - 1) for u in units)

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, row: dict, game_start_sec: float = 0.0, has_board: bool = False) -> np.ndarray:
        """
        Encode one state row.

        Args:
            row: A cleaned state row (dict) — see scripts/clean_states.py.
            game_start_sec: time_sec of the game's first frame.
            has_board: True if the champion detector produced board/bench for this game.
        """
        g = self.game
        board_units, bench_units = row.get("board") or [], row.get("bench") or []
        board = [u["name"] for u in board_units]
        bench = [u["name"] for u in bench_units]
        slots = (row.get("shop_slots") or []) + [None] * SHOP_SLOTS
        slots = slots[:SHOP_SLOTS]
        shop = [n for n in slots if n]

        owned = set(board) | set(bench)
        board_traits = set(g.trait_counts(board, board_units))
        tiers, active = self._trait_tiers(board, board_units)
        level = row.get("level") or 0
        # Team size (level, traits, augments, team-size items anywhere: GameData.team_size) minus slots used.
        elsewhere = [i for u in bench_units for i in u.get("items") or []] + [i for i in row.get("bench_items") or [] if i]
        team_size = g.team_size(level, board_units, elsewhere, row.get("augments") or []) if level else 0
        slots_used = sum(g.slots(n) for n in board)

        parts = [
            self._scalars(row, game_start_sec),
            self._lobby(row),
            [g.cost(n) / MAX_COST if n else 0.0 for n in slots],
            self._champion_counts(shop, SHOP_SLOTS),
            self._cost_mix(shop, SHOP_SLOTS),
            self._trait_vector(shop),
            [len(shop) / SHOP_SLOTS,
             _clip(sum(g.cost(n) for n in shop), SHOP_SLOTS * MAX_COST),
             sum(1 for n in shop if n in owned) / SHOP_SLOTS,
             sum(1 for n in shop if n in g.champions
                 and board_traits & set(g.champions[n].traits)) / SHOP_SLOTS],
            self._champion_counts(board, 3, units=True),
            self._cost_mix(board, MAX_BOARD),
            self._trait_vector(board, board_units),
            tiers,
            [_clip(len(board), MAX_BOARD),
             _clip(team_size - slots_used, MAX_LEVEL) if has_board else 0.0,
             _clip(self._value(board_units), 150),
             _clip(active, 10)],
            self._champion_counts(bench, 3, units=True),
            [_clip(len(bench), MAX_BENCH), _clip(self._value(bench_units), 100)],
            [_clip(len(row.get("augments") or []), MAX_AUGMENTS)]
            + [_clip((row.get("augment_tiers") or []).count(t), MAX_AUGMENTS) for t in AUGMENT_TIERS],
            [float(bool(row.get("shop_visible"))),
             float(bool(row.get("other_hp"))),
             float(has_board),
             float(bool(row.get("home_board"))),
             float(bool(row.get("planning")))],
        ]
        parts.insert(-1, self._items(row.get("bench_items") or []))
        vec = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
        assert vec.shape == (self.size,)
        return vec

    def encode_game(self, rows: list[dict]) -> np.ndarray:
        """Encode one game's rows (in frame order) into an array of shape [frames, size]."""
        if not rows:
            return np.zeros((0, self.size), dtype=np.float32)
        start = rows[0].get("time_sec", 0.0)
        has_board = any(r.get("board") or r.get("bench") for r in rows)
        return np.stack([self.encode(r, start, has_board) for r in rows])

    def describe(self) -> list[tuple[str, int, int]]:
        """(block name, first index, length) for each block of the vector."""
        out, pos = [], 0
        for name, names in self.blocks:
            out.append((name, pos, len(names)))
            pos += len(names)
        return out
