"""
Static knowledge about the current TFT set: each champion's cost and traits,
and each trait's activation breakpoints.

Loaded from assets/game_data/<set>.json (written by scripts/fetch_game_data.py).
The champion and trait orderings defined here fix the layout of the encoded
state vector, so they are sorted by name and never depend on what was seen.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.utils.config import vision_cfg

_ROOT = Path(__file__).resolve().parents[2]
MAX_COST = 5
BASIC_COMPONENTS = {
    "B.F. Sword", "Recurve Bow", "Needlessly Large Rod", "Tear Of The Goddess", "Chain Vest",
    "Negatron Cloak", "Giant's Belt", "Sparring Gloves", "Spatula", "Frying Pan",
}


@dataclass(frozen=True)
class Champion:
    name: str
    cost: int
    traits: tuple[str, ...]


class GameData:
    def __init__(self, raw: dict, items: dict | None = None):
        self.set_name: str = raw["set"]
        self.items: dict = items or {}

        # Form variants ("Lux (Coven)") share one shop card and one unit slot;
        # keep the base entry so names match what OCR and the detector produce.
        merged: dict[str, Champion] = {}
        for name, c in raw["champions"].items():
            base = name.split(" (")[0]
            if base not in merged or name == base:
                merged[base] = Champion(base, c["cost"], tuple(c["traits"]))
        self.champions: dict[str, Champion] = dict(sorted(merged.items()))
        self.champion_names: list[str] = list(self.champions)
        self.champion_index: dict[str, int] = {n: i for i, n in enumerate(self.champion_names)}

        self.breakpoints: dict[str, list[int]] = {
            name: t["breakpoints"] for name, t in sorted(raw["traits"].items())
        }
        self.trait_names: list[str] = list(self.breakpoints)
        self.trait_index: dict[str, int] = {n: i for i, n in enumerate(self.trait_names)}

    def cost(self, name: str) -> int:
        """Gold cost of a champion, 0 if the name is unknown."""
        c = self.champions.get(name)
        return c.cost if c else 0

    def item_kind(self, name: str | None) -> str:
        """'component', 'completed' (built from two components), or 'other' (also unknown icons)."""
        info = self.items.get(name) if name else None
        if not info:
            return "other"
        if info.get("recipe"):
            return "completed"
        return "component" if name in BASIC_COMPONENTS else "other"

    def trait_counts(self, names: list[str]) -> dict[str, int]:
        """Units per trait. Each distinct champion counts once, as in the game."""
        counts: dict[str, int] = {}
        for name in set(names):
            c = self.champions.get(name)
            for trait in (c.traits if c else ()):
                counts[trait] = counts.get(trait, 0) + 1
        return counts

    def active_tier(self, trait: str, count: int) -> int:
        """How many of the trait's breakpoints are reached (0 = inactive)."""
        return sum(1 for b in self.breakpoints.get(trait, []) if count >= b)


@lru_cache(maxsize=None)
def load_game_data(set_name: str | None = None) -> GameData:
    set_name = set_name or vision_cfg["game_data"]["set"]
    path = _ROOT / "assets" / "game_data" / f"{set_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/fetch_game_data.py")
    items_path = path.with_name(f"{set_name}_items.json")
    items = json.loads(items_path.read_text(encoding="utf-8")) if items_path.exists() else {}
    return GameData(json.loads(path.read_text(encoding="utf-8")), items)
