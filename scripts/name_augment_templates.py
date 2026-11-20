"""
Name the augment templates (assets/templates/augments/aug_XXX.png) from the
per-game augment labels (scripts/label_augments.py).

Usage:
    python scripts/name_augment_templates.py      # writes assets/templates/augment_names.json

In each labelled game, every own-board reading of bar slot k (a template ID
and the tier read from the icon's colour) is paired with the label for slot
k. A template whose pairings agree on one augment (at least MIN_READS
readings, AGREE of them the same) takes that name. Some templates are shared
by two augments that differ only by tier colour (Silver / Gold Destiny,
Ascension / Partial Ascension: the icon matches both); where a template's
votes split, it is named per tier instead, if each tier agrees on its own.
That lifted the held-out naming of already-labelled augments from 88% to 96%
(scripts/eval_augments.py --tier, 2026-11-19).

The result is a lookup, not renamed files: template IDs stay stable, so new
labels only need this script re-run, then scripts/clean_states.py.
    {"aug_010": "Ascension", ...,  "aug_005@silver": "Silver Destiny", "aug_005@gold": "Gold Destiny"}
Read by src/state/sequence.py (augment_name) when deciding a game's augments.
Templates never read in a labelled game stay unnamed (their ID is kept and
counts as "_unknown" in the token table).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter, defaultdict

from src.state.game_data import GameData
from src.utils import paths
from src.vision.arena import home_board_rows

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
TEMPLATE_LABELS = paths.REPO / "data" / "processed" / "augment_template_labels.json"   # label_new_augments.py
OUT = paths.REPO / "assets" / "templates" / "augment_names.json"
MIN_READS = 3
AGREE = 0.8


def agreed(c: Counter) -> str | None:
    name, n = c.most_common(1)[0]
    total = sum(c.values())
    return name if total >= MIN_READS and n / total >= AGREE else None


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    if not labels:
        sys.exit("No augment labels yet - run scripts/label_augments.py")
    votes: dict[tuple[str, str | None], Counter] = defaultdict(Counter)    # (template, tier) -> names
    for key, lab in labels.items():
        vod, game = key.split("/")
        path = paths.STATES / vod / f"{game}.jsonl"
        if not path.exists():
            continue
        rows = sorted((json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        for r, home in zip(rows, home_board_rows(rows)):
            if not home:
                continue
            tiers = r.get("augment_tiers") or []
            for k, tid in enumerate(r.get("augments") or []):
                if tid and k < len(lab["augments"]) and lab["augments"][k] != "_unknown":
                    # "+" versions share the icon: one name (GameData.augment_label)
                    votes[(tid, tiers[k] if k < len(tiers) else None)][GameData.augment_label(lab["augments"][k])] += 1
    by_tid: dict[str, Counter] = defaultdict(Counter)
    for (tid, _), c in votes.items():
        by_tid[tid].update(c)

    names, by_tier, unsure = {}, 0, {}
    for tid, c in sorted(by_tid.items()):
        if (name := agreed(c)):
            names[tid] = name
            continue
        split = {tier: agreed(cc) for (t, tier), cc in votes.items() if t == tid}
        for tier, name in split.items():
            if name:
                names[f"{tid}@{tier}"] = name
                by_tier += 1
        if not any(split.values()):
            unsure[tid] = dict(c.most_common(3))
    # Names Hamza gave a template directly (scripts/label_new_augments.py) come before the votes;
    # "_junk" leaves it unnamed.
    direct = json.loads(TEMPLATE_LABELS.read_text(encoding="utf-8")) if TEMPLATE_LABELS.exists() else {}
    for tid, name in direct.items():
        for key in [k for k in names if k == tid or k.startswith(tid + "@")]:
            del names[key]
        unsure.pop(tid, None)
        if name and not name.startswith("_"):
            names[tid] = GameData.augment_label(name)
    if direct:
        print(f"{sum(1 for n in direct.values() if n and not n.startswith('_'))} templates named directly, "
              f"{sum(1 for n in direct.values() if n == '_junk')} marked junk")
    OUT.write_text(json.dumps(names, indent=1, sort_keys=True), encoding="utf-8")
    n_tid = sum(1 for k in names if "@" not in k)
    print(f"{len(labels)} labelled games: {n_tid} templates named, {by_tier} more names by tier, "
          f"{len(unsure)} unclear -> {OUT}")
    for tid, c in list(unsure.items())[:15]:
        print(f"  unclear {tid}: {c}")


if __name__ == "__main__":
    main()
