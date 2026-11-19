"""
Download the official icon of every augment in the set (Community Dragon), for
naming the augment templates cut from footage, and save the set's augment list.

Usage:
    python scripts/fetch_augment_icons.py
    python scripts/fetch_augment_icons.py --from-file path/to/en_us.json

Output:
    assets/templates/augments_official/<Augment Name>__<tier>.png   (64 px; tier silver/gold/prismatic)
    assets/game_data/<set>_augments.json   name -> {api_names, tiers}

The tier comes from the icon's file name (..._i / _ii / _iii). One augment name
can exist at several tiers (Worth the Wait: gold and prismatic).
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

URL = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
GAME = "https://raw.communitydragon.org/latest/game/"
OUT = Path("assets/templates/augments_official")
TIERS = {"i": "silver", "ii": "gold", "iii": "prismatic"}
HEADERS = {"User-Agent": "Mozilla/5.0 (tft-project)"}


def tier_of(icon: str) -> str | None:
    m = re.search(r"_(i{1,3})(?:_[a-z0-9]+)?\.(?:tex|png)$", icon.lower())
    return TIERS.get(m.group(1)) if m else None


def fetch(job):
    url, dest = job
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 404:
                return dest, False
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return dest, False
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[..., 3:4].astype(np.float32) / 255
                img = (img[..., :3].astype(np.float32) * alpha).astype(np.uint8)
            cv2.imwrite(str(dest), cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))
            return dest, True
        except requests.RequestException:
            continue
    return dest, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=vision_cfg["game_data"]["set"])
    parser.add_argument("--from-file")
    args = parser.parse_args()
    if args.from_file:
        raw = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
    else:
        raw = requests.get(URL, timeout=300, headers=HEADERS).json()
    data = next(s for s in raw["setData"] if s.get("mutator") == args.set)
    items = {it["apiName"]: it for it in raw["items"]}

    augments: dict[str, dict] = {}
    jobs, seen = [], set()
    OUT.mkdir(parents=True, exist_ok=True)
    for api in sorted(data.get("augments") or []):
        it = items.get(api)
        if not it or not it.get("name"):
            continue
        icon = it.get("icon") or ""
        tier = tier_of(icon)
        entry = augments.setdefault(it["name"], {"api_names": [], "tiers": []})
        entry["api_names"].append(api)
        if tier and tier not in entry["tiers"]:
            entry["tiers"].append(tier)
        if icon and "missing" not in icon and icon not in seen:
            seen.add(icon)
            safe = it["name"].replace("/", "-").replace(":", "")
            url = GAME + re.sub(r"\.tex$", ".png", icon.lower())
            jobs.append((url, OUT / f"{safe}__{tier or 'unknown'}.png"))

    out = Path(f"assets/game_data/{args.set}_augments.json")
    out.write_text(json.dumps(dict(sorted(augments.items())), indent=1), encoding="utf-8")
    with ThreadPoolExecutor(4) as pool:
        got = sum(ok for _, ok in pool.map(fetch, jobs))
    print(f"{len(augments)} augments -> {out}; {got}/{len(jobs)} icons -> {OUT}")


if __name__ == "__main__":
    main()
