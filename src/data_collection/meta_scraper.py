"""
Scrape top TFT compositions from MetaTFT / Tactics.tools.
Outputs structured JSON to data/processed/meta_comps/.
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


METATFT_URL = "https://www.metatft.com/comps"
HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}


def scrape_metatft(output_path: str, max_comps: int = 50) -> list[dict]:
    """
    Fetch top compositions from MetaTFT and save as JSON.

    Returns list of comp dicts with keys:
      name, tier, avg_placement, play_rate, units, carry_items, augments
    """
    # TODO: MetaTFT is JS-heavy — replace requests+BS4 with Selenium or
    # Playwright if the page doesn't render statically.
    resp = requests.get(METATFT_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    comps = _parse_comps(soup, max_comps)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(comps, f, indent=2)

    return comps


def _parse_comps(soup: BeautifulSoup, limit: int) -> list[dict]:
    # Placeholder — DOM selectors depend on the live page structure.
    # Update these selectors after inspecting MetaTFT's rendered HTML.
    comps = []
    comp_cards = soup.select(".comp-card")[:limit]
    for card in comp_cards:
        comp = {
            "name": card.select_one(".comp-name").get_text(strip=True) if card.select_one(".comp-name") else "Unknown",
            "tier": card.get("data-tier", ""),
            "avg_placement": float(card.get("data-avg-placement", 0)),
            "play_rate": float(card.get("data-play-rate", 0)),
            "units": [],
            "carry_items": [],
            "augments": [],
        }
        comps.append(comp)
    return comps
