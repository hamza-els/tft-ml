"""
Static knowledge about the current TFT set: each champion's cost and traits,
each trait's activation breakpoints, emblems, and what changes the board's
team size (traits, multi-slot champions, team-size items).

Loaded from assets/game_data/<set>.json (written by scripts/fetch_game_data.py).
The champion and trait orderings defined here fix the layout of the encoded
state vector, so they are sorted by name and never depend on what was seen.
"""

import json
import re
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
    def __init__(self, raw: dict, items: dict | None = None, extra_units: list[str] | None = None,
                 augments: dict | None = None):
        self.set_name: str = raw["set"]
        self.items: dict = items or {}
        self.augments: dict = augments or {}         # name -> {api_names, tiers} (fetch_augment_icons.py)

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

        # Everything that can stand on the board or bench: shop champions plus
        # trait/augment units (Stonebark Tree, Training Dummy, ...), which have
        # no cost and no traits.
        self.extra_units: list[str] = sorted(set(extra_units or []) - set(self.champions))
        self.unit_names: list[str] = sorted(self.champion_names + self.extra_units)
        self.unit_index: dict[str, int] = {n: i for i, n in enumerate(self.unit_names)}

        # What changes the board's team size (from the set's own descriptions,
        # see scripts/fetch_game_data.py). Trait/augment units take no slot.
        self.trait_team_size: dict[str, list[int]] = {
            name: t["team_size"] for name, t in raw["traits"].items() if t.get("team_size")}
        self.team_slots: dict[str, int] = {}
        self.grants_trait: dict[str, dict[str, int]] = {}
        for name, c in raw["champions"].items():
            base = name.split(" (")[0]
            if c.get("team_slots"):
                self.team_slots[base] = c["team_slots"]
            if c.get("grants_trait"):
                self.grants_trait[base] = c["grants_trait"]
        self.team_size_items: dict[str, int] = raw.get("team_size_items") or {}
        self.emblems: dict[str, str] = raw.get("emblems") or {}

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

    def trait_counts(self, names: list[str], units: list[dict] | None = None) -> dict[str, int]:
        """
        Units per trait. Each distinct champion counts once, as in the game.

        With `units` (board units with their "items"), emblems count too: a
        unit holding "<Trait> Emblem" adds that trait if it doesn't have it.
        Champions that grant extra trait count (Elder Dragon: +2 Riftbeast)
        add it once.
        """
        counts: dict[str, int] = {}
        for name in set(names):
            c = self.champions.get(name)
            for trait in (c.traits if c else ()):
                counts[trait] = counts.get(trait, 0) + 1
            for trait, n in self.grants_trait.get(name, {}).items():
                counts[trait] = counts.get(trait, 0) + n
        for u in units or []:
            c = self.champions.get(u.get("name"))
            own = set(c.traits) if c else set()
            for item in set(i for i in (u.get("items") or []) if i):
                trait = self.emblems.get(item)
                if trait and trait not in own:
                    counts[trait] = counts.get(trait, 0) + 1
                    own.add(trait)
        return counts

    def active_tier(self, trait: str, count: int) -> int:
        """How many of the trait's breakpoints are reached (0 = inactive)."""
        return sum(1 for b in self.breakpoints.get(trait, []) if count >= b)

    def item_base(self, name: str | None) -> tuple[str | None, bool]:
        """(base item, radiant): "Radiant Warmog's Armor" -> ("Warmog's Armor", True)."""
        if name and name.startswith("Radiant ") and name[len("Radiant "):] in self.items:
            return name[len("Radiant "):], True
        return name, False

    @staticmethod
    def augment_base(name: str | None) -> str | None:
        """
        The augment an augment name is a tier of: "New Recruit+" -> "New Recruit",
        "Celestial Blessing II" -> "Celestial Blessing" (Hamza, 2026-11-19: tiers
        of one augment are variations, not different augments).
        """
        if not name:
            return name
        return re.sub(r"\s*(\+|\b(?:I{1,3}))$", "", name).strip()

    @staticmethod
    def augment_label(name: str | None) -> str | None:
        """
        The name an augment is labelled under: "+" versions folded in ("Money Hungry+",
        "Booster Pack++" -> "Money Hungry", "Booster Pack"). They share the icon, so a
        frame can't tell them apart (Hamza, 2026-11-19). Unlike augment_base, roman-numeral
        tiers (I/II/III) stay apart here.
        """
        return name.rstrip("+").rstrip() if name else name

    def augment_vocab(self) -> list[str]:
        """Augment IDs: 0 = padding, every augment of the set by base name, then unknown."""
        return ["<pad>"] + sorted({self.augment_base(n) for n in self.augments}) + ["_unknown"]

    def unit_vocab(self) -> list[str]:
        """Champion IDs for the token table: 0 = padding, then every unit of the set, then unknown.
        Fixed by the set data, so IDs never shift with what was seen."""
        return ["<pad>"] + self.unit_names + ["_unknown"]

    def item_vocab(self) -> list[str]:
        """Item IDs: 0 = padding, every item of the set with Radiant versions folded into their
        base item (the radiant flag is a separate field), then unknown."""
        names = {self.item_base(n)[0] for n in self.items} | set(self.emblems)
        return ["<pad>"] + sorted(names) + ["_unknown"]

    def slots(self, name: str) -> int:
        """Team slots a unit takes on the board: 0 for trait/augment units, 2 for Elder Dragon, else 1."""
        if name in self.extra_units:
            return 0
        return self.team_slots.get(name, 1)

    def team_size_bonus(self, units: list[dict]) -> int:
        """
        Max team size above the player's level that the board itself shows:
        active traits that add team size (Riftbeast 10: +2) and team-size
        items held by units (Tactician's Crown: +1). Augments that add team
        size (New Recruit, Cursed Crown, ...) are not visible here.
        """
        counts = self.trait_counts([u["name"] for u in units], units)
        bonus = 0
        for trait, sizes in self.trait_team_size.items():
            tier = self.active_tier(trait, counts.get(trait, 0))
            if tier:
                bonus += sizes[tier - 1]
        for u in units:
            bonus += sum(self.team_size_items.get(i, 0) for i in (u.get("items") or []) if i)
        return bonus


@lru_cache(maxsize=None)
def load_game_data(set_name: str | None = None) -> GameData:
    set_name = set_name or vision_cfg["game_data"]["set"]
    path = _ROOT / "assets" / "game_data" / f"{set_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/fetch_game_data.py")
    items_path = path.with_name(f"{set_name}_items.json")
    items = json.loads(items_path.read_text(encoding="utf-8")) if items_path.exists() else {}
    extra = vision_cfg["game_data"].get("extra_units") or []
    aug_path = path.with_name(f"{set_name}_augments.json")
    augments = json.loads(aug_path.read_text(encoding="utf-8")) if aug_path.exists() else {}
    return GameData(json.loads(path.read_text(encoding="utf-8")), items, extra, augments)
