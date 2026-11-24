"""
Static knowledge about the current TFT set: each champion's cost and traits,
each trait's activation breakpoints, emblems, and what changes the board's
team size (traits, multi-slot champions, team-size items).

Loaded from assets/game_data/<set>.json (written by scripts/fetch_game_data.py).
The champion and trait orderings defined here fix the layout of the encoded
state vector, so they are sorted by name and never depend on what was seen.
"""

import json
from collections import Counter
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


_BUNDLE = re.compile(r"^\d+\s+(.+)$")


def fold_item_bundles(items: dict) -> tuple[dict, dict[str, str | None]]:
    """
    The set's item list also holds loot-table entries whose name starts with a
    number: bundles of one item ("2 Reforgers", "6 Lesser Champion Duplicators",
    same icon as the single item) and rewards that are no item at all ("10 gold",
    "2-star 3-cost champion", "4 components"). The item column only ever shows
    single items, so bundles fold into their item and the rest is dropped.
    Returns (items without those entries, {entry: single item, or None if dropped}).
    """
    kept, aliases = {}, {}
    for name, info in items.items():
        if not name[:1].isdigit():
            kept[name] = info
            continue
        single = None
        m = _BUNDLE.match(name)
        if m:
            rest = m.group(1)
            for cand in (rest, rest[:-1], re.sub(r"s' ", "'s ", rest)[:-1]):   # Tacticians' Capes
                if cand in items and not cand[:1].isdigit():
                    single = cand
                    break
        aliases[name] = single
    return kept, aliases


@dataclass(frozen=True)
class Champion:
    name: str
    cost: int
    traits: tuple[str, ...]


class GameData:
    def __init__(self, raw: dict, items: dict | None = None, extra_units: list[str] | None = None,
                 augments: dict | None = None, unit_aliases: dict[str, str] | None = None):
        self.set_name: str = raw["set"]
        # Bundles ("2 Reforgers") fold into their item; loot rewards ("10 gold") are no items.
        self.items, self.item_aliases = fold_item_bundles(items or {})
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
        # Units drawn exactly like another one are folded into it (unit_aliases in the config:
        # Small Furry Friend -> Big Furry Friend), so they share one name and one ID.
        self.unit_aliases: dict[str, str] = dict(unit_aliases or {})
        self.extra_units: list[str] = sorted(set(extra_units or []) - set(self.champions)
                                             - set(self.unit_aliases))
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
        # Augments that raise team size (New Recruit +1, Cursed Crown +2) and augments that hand
        # out a team-size item at once (Coronation: Tactician's Crown); scripts/fetch_game_data.py.
        self.team_size_augments: dict[str, int] = raw.get("team_size_augments") or {}
        self.augment_items: dict[str, str] = raw.get("augment_items") or {}
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

    def canonical_unit(self, name: str | None) -> str | None:
        """'Small Furry Friend' -> 'Big Furry Friend' (same picture); any other name unchanged."""
        return self.unit_aliases.get(name, name) if name else name

    def canonical_item(self, name: str | None) -> str | None:
        """'2 Reforgers' -> 'Reforger'; '10 gold' -> None (no item); any other name unchanged."""
        if name in self.item_aliases:
            return self.item_aliases[name]
        return name

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
        # every trailing "+" ("Invested++", "Pandora's Items II+"), then the tier numeral
        return re.sub(r"\s*\b(?:I{1,3})$", "", name.rstrip("+").rstrip()).strip()

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

    def item_team_size(self, items) -> int:
        """Team size from team-size items (Tactician's Crown/Cape/Shield) among `items`."""
        return sum(self.team_size_items.get(i, 0) for i in items if i)

    def team_size(self, level: int, board: list[dict], items_elsewhere=(), augments=(),
                  kept_items: dict[str, int] | None = None) -> int:
        """
        Max team size worked out from the state (Hamza, 2026-11-24), instead of read off the
        arena: level + active team-size traits (Riftbeast 10: +2) + team-size augments (New
        Recruit +1, Cursed Crown +2) + team-size items. The items count wherever they are:
        held by board or bench units or lying in the item column (`items_elsewhere`); their
        text is "your team gains +1 max team size", and on footage the counter shows the +1
        with a Tactician's Cape in the column. `kept_items` ({item: copies}) are items known
        to be owned from earlier frames (they can't disappear), counted when more than seen
        now; an augment that hands one out (Coronation) counts as owning it.
        Against the counter's own team-size number on 3,984 rounds of 202 games (augments from
        Hamza's labels, items kept once seen on 3 frames of a round): 98.4% equal; level +
        traits + board items alone 96.1%.
        """
        counts = self.trait_counts([u["name"] for u in board], board)
        size = int(level)
        for trait, sizes in self.trait_team_size.items():
            tier = self.active_tier(trait, counts.get(trait, 0))
            if tier:
                size += sizes[tier - 1]
        size += sum(self.team_size_augments.get(a, 0) for a in augments if a)
        held = Counter(i for u in board for i in (u.get("items") or []) if i in self.team_size_items)
        held.update(i for i in items_elsewhere if i in self.team_size_items)
        owned = Counter(kept_items or {})
        for a in augments:
            if a in self.augment_items:
                owned[self.augment_items[a]] = max(owned[self.augment_items[a]], 1)
        items = held | owned                                     # per item, the larger count
        return size + sum(self.team_size_items[i] * n for i, n in items.items())

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
    # Trait-granted items with no official entry (configs/vision_config.yaml).
    for name in vision_cfg["game_data"].get("extra_items") or []:
        items.setdefault(name, {"api_name": None, "kind": "trait", "recipe": []})
    extra = vision_cfg["game_data"].get("extra_units") or []
    aug_path = path.with_name(f"{set_name}_augments.json")
    augments = json.loads(aug_path.read_text(encoding="utf-8")) if aug_path.exists() else {}
    aliases = vision_cfg["game_data"].get("unit_aliases") or {}
    return GameData(json.loads(path.read_text(encoding="utf-8")), items, extra, augments, aliases)
