"""
Apply the item review (from data/raw/review/item_review.html) to the
templates in assets/templates/unit_items/, so each item has one ID.

Usage:
    python scripts/apply_item_review.py data/raw/review/item_review_result.json --dry-run
    python scripts/apply_item_review.py data/raw/review/item_review_result.json

- "same" pairs are joined (transitively) into one item.
- "names" give an unnamed ID a name; typed names are matched to the
  official item names (assets/game_data/<set>_items.json and the names
  already used by templates), e.g. "Warmog's" -> "Warmog's Armor".
- A group's name: a name Hamza typed beats a name the harvest gave
  automatically (that one comes from matching against official art, which
  can't tell Radiant from normal at 25 px); between automatic names the
  shortest (the non-Radiant one) wins; a group with no name keeps its
  lowest uitem_ ID.
- Files are renamed to Name.png, Name__2.png, ...; "different" pairs that
  end up in one group are reported and the run stops.

Note: scripts/harvest_unit_item_icons.py rebuilds the folder from scratch
and renumbers uitem_ IDs, which would undo this. Re-harvest only on purpose,
then review again.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import difflib
import json
import shutil

from src.state.game_data import load_game_data
from src.utils.config import vision_cfg
from src.vision.items import _ROOT

FOLDER = _ROOT / vision_cfg["unit_bar"]["item_templates"]


def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def official(name: str, known: list[str]) -> str:
    """The known item name a typed name means (prefix, then spelling similarity)."""
    q = norm(name)
    radiant = q.startswith("radiant")
    pool = [k for k in known if norm(k).startswith("radiant") == radiant]
    best, score = name, 0.0
    for k in pool:
        n = norm(k)
        s = difflib.SequenceMatcher(None, q, n).ratio()
        if n.startswith(q) or q.startswith(n):
            s += 1.0
        elif n.startswith(q[:6]):
            s += 0.5
        if s > score:
            best, score = k, s
    return best if score >= 0.6 else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="JSON pasted from the review page")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    review = json.loads(Path(args.result).read_text(encoding="utf-8"))

    files = sorted(FOLDER.glob("*.png"))
    ids = sorted({f.stem.split("__")[0] for f in files})
    game = load_game_data()
    known = sorted(set(game.items) | {i for i in ids if not i.startswith("uitem_")})

    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in review.get("same", []):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    bad = [(a, b) for a, b in review.get("different", []) if a in parent and b in parent and find(a) == find(b)]
    if bad:
        sys.exit(f"'different' pairs ended up joined through other 'same' answers: {bad}")

    typed = {k: official(v, known) for k, v in review.get("names", {}).items()}
    final: dict[str, str] = {}
    for members in groups.values():
        names = [typed[m] for m in members if m in typed]
        auto = sorted((m for m in members if not m.startswith("uitem_")), key=len)
        name = (max(set(names), key=names.count) if names else auto[0] if auto else sorted(members)[0])
        for m in members:
            final[m] = name

    changes = {m: n for m, n in final.items() if m != n}
    print("Typed names matched to:", {k: f"{review['names'][k]} -> {v}" for k, v in typed.items()})
    print(f"{len(changes)} IDs renamed:")
    for m, n in sorted(changes.items()):
        print(f"  {m:28s} -> {n}")
    print(f"{len(ids)} IDs before, {len(set(final.values()))} after")
    if args.dry_run:
        return

    backup = FOLDER.parent / (FOLDER.name + "_before_review")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(FOLDER, backup)
    staged = []
    for f in files:
        staged.append((f.read_bytes(), final[f.stem.split("__")[0]]))
    for f in files:
        f.unlink()
    used: dict[str, int] = {}
    for data, name in staged:
        used[name] = used.get(name, 0) + 1
        stem = name if used[name] == 1 else f"{name}__{used[name]}"
        (FOLDER / f"{stem}.png").write_bytes(data)
    print(f"Renamed; old files kept in {backup}")


if __name__ == "__main__":
    main()
