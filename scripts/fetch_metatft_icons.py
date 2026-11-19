"""
Download the item icons metatft.com shows (cdn.metatft.com) for every item of
the set, as a second reference art set next to Community Dragon's
(assets/templates/items/, scripts/fetch_item_icons.py). Hamza found the
Community Dragon art "a bit off on some details" next to the in-game icons.

Usage:
    python scripts/fetch_metatft_icons.py

Output: assets/templates/items_metatft/<Item Name>.png (64 px), for every item
in assets/game_data/<set>_items.json whose icon the CDN has
(https://cdn.metatft.com/file/metatft/items/<api name, lowercase>.png).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests

from src.utils.config import vision_cfg

URL = "https://cdn.metatft.com/file/metatft/items/{}.png"
OUT = Path("assets/templates/items_metatft")
SIZE = 64


def fetch(item):
    name, api = item
    for attempt in range(3):
        try:
            r = requests.get(URL.format(api.lower()), timeout=30, headers={"User-Agent": "Mozilla/5.0 (tft-project)"})
            if r.status_code == 404:
                return name, None
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return name, None
            if img.ndim == 3 and img.shape[2] == 4:          # transparent background -> black
                alpha = img[..., 3:4].astype(np.float32) / 255
                img = (img[..., :3].astype(np.float32) * alpha).astype(np.uint8)
            return name, cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        except requests.RequestException:
            time.sleep(2)
    return name, None


def main():
    set_name = vision_cfg["game_data"]["set"]
    items = json.loads(Path(f"assets/game_data/{set_name}_items.json").read_text(encoding="utf-8"))
    todo = sorted((name, info["api_name"]) for name, info in items.items() if info.get("api_name"))
    OUT.mkdir(parents=True, exist_ok=True)
    got = missing = 0
    with ThreadPoolExecutor(4) as pool:
        for name, img in pool.map(fetch, todo):
            if img is None:
                missing += 1
                continue
            safe = name.replace("/", "-")
            cv2.imwrite(str(OUT / f"{safe}.png"), img)
            got += 1
    print(f"{got} icons -> {OUT}  ({missing} not on the CDN)")


if __name__ == "__main__":
    main()
