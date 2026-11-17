"""
Download the official icon of every item in the current set (Community Dragon)
into assets/templates/items/, for matching against in-game item icons.

Usage:
    python scripts/fetch_item_icons.py
    python scripts/fetch_item_icons.py --set TFTSet18 --size 64

Output:
    assets/templates/items/<Item Name>.png     (square, --size px)
    assets/game_data/<set>_items.json          name -> api name, kind, recipe

Several internal entries share one icon and name (old and new api names for
the same item); each distinct name is saved once. Augments are skipped.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from src.utils.config import vision_cfg

DATA_URL = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
ICON_BASE = "https://raw.communitydragon.org/latest/game/"
HEADERS = {"User-Agent": "Mozilla/5.0 (tft-project)"}   # the server rejects python's default


def kind_of(item: dict) -> str:
    api = item["apiName"]
    if "Component" in api or (api.startswith("TFT_Item_") and not item.get("composition")
                              and "Artifact" not in api and "Radiant" not in api):
        return "component_or_special"
    if "Artifact" in api:
        return "artifact"
    if "Emblem" in api:
        return "emblem"
    if "Radiant" in api:
        return "radiant"
    if item.get("composition"):
        return "completed"
    return "other"


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=vision_cfg["game_data"]["set"])
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    print("Downloading set data ...")
    raw = requests.get(DATA_URL, timeout=300, headers=HEADERS).json()
    data = next((s for s in raw["setData"] if s.get("mutator") == args.set), None)
    if data is None:
        sys.exit(f"Set {args.set} not found")
    by_api = {i["apiName"]: i for i in raw["items"]}

    chosen: dict[str, dict] = {}
    for api in data["items"]:
        item = by_api.get(api)
        if not item or item.get("isAugment") or not item.get("name") or not item.get("icon"):
            continue
        if "Debug" in api or "<" in item["name"]:
            continue
        name = safe_name(item["name"])
        # Prefer the set's own entry (DA_...) when two api names share a display name.
        if name not in chosen or api.startswith("DA_"):
            chosen[name] = item

    out_dir = Path(vision_cfg["template_matching"]["item_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(entry: tuple[str, dict]) -> str | None:
        name, item = entry
        url = ICON_BASE + item["icon"].lower().replace(".tex", ".png").replace(".dds", ".png")
        try:
            resp = requests.get(url, timeout=60, headers=HEADERS)
            resp.raise_for_status()
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return name
            img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", img)
            (out_dir / f"{name}.png").write_bytes(buf.tobytes())
            return None
        except requests.RequestException:
            return name

    with ThreadPoolExecutor(8) as pool:
        failed = [n for n in pool.map(fetch, chosen.items()) if n]

    meta = {
        name: {"api_name": item["apiName"], "kind": kind_of(item),
               "recipe": [by_api[c]["name"] for c in item.get("composition") or [] if c in by_api]}
        for name, item in sorted(chosen.items()) if name not in failed
    }
    meta_path = Path("assets/game_data") / f"{args.set}_items.json"
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"{len(meta)} item icons -> {out_dir}")
    print(f"metadata -> {meta_path}")
    if failed:
        print(f"{len(failed)} failed to download: {failed[:15]}")


if __name__ == "__main__":
    main()
