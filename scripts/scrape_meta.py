"""
CLI script: scrape top TFT comps from MetaTFT and save to data/processed/meta_comps/.

Usage:
    python scripts/scrape_meta.py [--max-comps 50]
"""

import argparse
from pathlib import Path
from src.data_collection.meta_scraper import scrape_metatft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-comps", type=int, default=50)
    args = parser.parse_args()

    out_path = Path("data/processed/meta_comps/metatft.json")
    comps = scrape_metatft(str(out_path), max_comps=args.max_comps)
    print(f"Saved {len(comps)} comps to {out_path}")


if __name__ == "__main__":
    main()
