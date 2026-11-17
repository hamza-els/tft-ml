"""
Download the official TFT set data (Community Dragon) and save the part the
project uses: every purchasable champion with its cost and traits, and every
trait with its activation breakpoints.

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
        breakpoints = sorted({e["minUnits"] for e in t["effects"] if e.get("minUnits")})
        if breakpoints:
            traits[t["name"]] = {"api_name": t["apiName"], "breakpoints": breakpoints}

    return {
        "set": set_key,
        "source": URL,
        "champions": dict(sorted(champions.items())),
        "traits": dict(sorted(traits.items())),
    }


def shop_names(champions: dict) -> list[str]:
    """Names as printed on shop cards: form variants like 'Lux (Coven)' collapse to 'Lux'."""
    return sorted({name.split(" (")[0] for name in champions})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=vision_cfg["game_data"]["set"])
    args = parser.parse_args()

    print(f"Downloading {URL} ...")
    # The server rejects the default python user agent.
    resp = requests.get(URL, timeout=300, headers={"User-Agent": "Mozilla/5.0 (tft-project)"})
    resp.raise_for_status()
    raw = resp.json()

    game = build(raw, args.set)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.set}.json"
    out.write_text(json.dumps(game, indent=1), encoding="utf-8")
    print(f"{len(game['champions'])} champions, {len(game['traits'])} traits -> {out}")

    names = shop_names(game["champions"])
    champ_list = Path(vision_cfg["ocr"]["champion_list"])
    champ_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"{len(names)} shop names -> {champ_list}")


if __name__ == "__main__":
    main()
