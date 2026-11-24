"""
Download the official TFT set data (Community Dragon) and save the part the
project uses: every purchasable champion with its cost and traits, every
trait with its activation breakpoints, and what changes the board's team
size: traits that add max team size at a breakpoint (Riftbeast 10: +2),
champions that take more than one slot (Elder Dragon: 2), items that add
team size while held (Tactician's Crown: +1). Also which emblem gives which
trait. Augments that add team size are not used: augment icons are only
known by ID.

Usage:
    python scripts/fetch_game_data.py                 # set from configs/vision_config.yaml
    python scripts/fetch_game_data.py --set TFTSet18

Output:
    assets/game_data/<set>.json
    assets/champions.txt   (shop-name list used by OCR; rewritten from the roster)

Note: the "latest" file reflects the current patch. Costs and traits can shift
between patches, so footage from an older patch may differ slightly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re

import requests

from src.utils.config import vision_cfg

URL = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
OUT_DIR = Path("assets/game_data")


def build(raw: dict, set_key: str) -> dict:
    sets = [s for s in raw["setData"] if s.get("mutator") == set_key]
    if not sets:
        known = sorted({s.get("mutator") for s in raw["setData"]})
        sys.exit(f"Set {set_key} not found. Available: {known}")
    data = sets[0]

    # Purchasable units have traits; trait-less entries are PvE monsters,
    # anvils and the like (some reuse a real unit's name, e.g. "Krug").
    champions = {}
    for c in data["champions"]:
        if not c["traits"] or not 1 <= c["cost"] <= 5:
            continue
        stats = c["stats"]
        champions[c["name"]] = {
            "api_name": c["apiName"],
            "cost": c["cost"],
            "traits": c["traits"],
            "hp": stats.get("hp"),
            "attack_damage": stats.get("damage"),
            "armor": stats.get("armor"),
            "range": stats.get("range"),
        }

    traits = {}
    for t in data["traits"]:
        effects = sorted((e for e in t["effects"] if e.get("minUnits")), key=lambda e: e["minUnits"])
        breakpoints = sorted({e["minUnits"] for e in effects})
        if not breakpoints:
            continue
        entry = {"api_name": t["apiName"], "breakpoints": breakpoints}
        # Extra max team size at a breakpoint ("+@TeamSize@ maximum team size", e.g. Riftbeast 10).
        team_size = [int((e.get("variables") or {}).get("TeamSize") or 0) for e in effects]
        if any(team_size):
            entry["team_size"] = team_size
        desc = re.sub(r"<[^>]+>", "", t.get("desc") or "")
        # "Elder Dragon takes up 2 team slots and grants +2 to the Riftbeast trait."
        if m := re.search(r"takes up (\d+) team slots", desc):
            entry["team_slots"] = int(m.group(1))
        if m := re.search(r"grants \+(\d+) to the ([\w' ]+?) trait", desc):
            entry["grants_trait"] = {m.group(2): int(m.group(1))}
        traits[t["name"]] = entry

    # Champions of a trait that "takes up N team slots" count N on the board.
    for c in champions.values():
        for tr in c["traits"]:
            if "team_slots" in traits.get(tr, {}):
                c["team_slots"] = traits[tr]["team_slots"]
            if "grants_trait" in traits.get(tr, {}):
                c["grants_trait"] = traits[tr]["grants_trait"]

    # Items that raise max team size while held (Tactician's Crown, ...), and
    # emblems: "<Trait> Emblem" gives its holder that trait.
    team_size_items, emblems = {}, {}
    for it in raw.get("items", []):
        name = it.get("name") or ""
        bonus = (it.get("effects") or {}).get("MaxArmySizeIncrease")
        if bonus and "Augment" not in (it.get("apiName") or ""):
            team_size_items[name] = int(bonus)
        if name.endswith(" Emblem") and name[:-len(" Emblem")] in traits:
            emblems[name] = name[:-len(" Emblem")]

    return {
        "set": set_key,
        "source": URL,
        "champions": dict(sorted(champions.items())),
        "traits": dict(sorted(traits.items())),
        "team_size_items": dict(sorted(team_size_items.items())),
        "emblems": dict(sorted(emblems.items())),
    }


def augment_team_size(raw: dict, augment_apis: set[str] | None) -> tuple[dict, dict]:
    """
    Augments that raise max team size ({name: +n}, "MaxArmySizeIncrease": New Recruit,
    Cursed Crown) and augments that hand out a team-size item at once ({name: item},
    "Gain a Tactician's Crown": Coronation; a delayed one, "After 3 rounds, gain ...",
    is left to the item readers). augment_apis: the set's augment api names
    (assets/game_data/<set>_augments.json), None = every augment.
    """
    sizes, gives = {}, {}
    items = {it.get("name") for it in raw.get("items", []) if (it.get("effects") or {}).get("MaxArmySizeIncrease")}
    for it in raw.get("items", []):
        api, name = it.get("apiName") or "", it.get("name") or ""
        if "Augment" not in api or (augment_apis is not None and api not in augment_apis):
            continue
        bonus = (it.get("effects") or {}).get("MaxArmySizeIncrease")
        if bonus:
            sizes[name] = int(bonus)
        m = re.match(r"\s*Gain an? ([^.<]+?)\.", re.sub(r"<[^>]+>", "", it.get("desc") or ""))
        if m and m.group(1) in items:
            gives[name] = m.group(1)
    return dict(sorted(sizes.items())), dict(sorted(gives.items()))


def shop_names(champions: dict) -> list[str]:
    """Names as printed on shop cards: form variants like 'Lux (Coven)' collapse to 'Lux'."""
    return sorted({name.split(" (")[0] for name in champions})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=vision_cfg["game_data"]["set"])
    parser.add_argument("--from-file", help="Use an already downloaded en_us.json instead of downloading")
    parser.add_argument("--team-size-only", action="store_true",
                        help="Only add the augment team-size fields to the existing <set>.json")
    args = parser.parse_args()

    if args.from_file:
        raw = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
    else:
        print(f"Downloading {URL} ...")
        # The server rejects the default python user agent.
        resp = requests.get(URL, timeout=300, headers={"User-Agent": "Mozilla/5.0 (tft-project)"})
        resp.raise_for_status()
        raw = resp.json()

    out = OUT_DIR / f"{args.set}.json"
    aug_path = OUT_DIR / f"{args.set}_augments.json"
    apis = ({a for v in json.loads(aug_path.read_text(encoding="utf-8")).values() for a in v["api_names"]}
            if aug_path.exists() else None)
    game = json.loads(out.read_text(encoding="utf-8")) if args.team_size_only else build(raw, args.set)
    game["team_size_augments"], game["augment_items"] = augment_team_size(raw, apis)
    print(f"team size augments {game['team_size_augments']}, augments giving team-size items {game['augment_items']}")
    if args.team_size_only:
        out.write_text(json.dumps(game, indent=1), encoding="utf-8")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(game, indent=1), encoding="utf-8")
    print(f"{len(game['champions'])} champions, {len(game['traits'])} traits -> {out}")

    names = shop_names(game["champions"])
    champ_list = Path(vision_cfg["ocr"]["champion_list"])
    champ_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} shop names -> {champ_list}")


if __name__ == "__main__":
    main()
