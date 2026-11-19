"""
Name the augment templates (assets/templates/augments/aug_XXX.png) from the
per-game augment labels (scripts/label_augments.py).

Usage:
    python scripts/name_augment_templates.py            # writes data/raw/review/augment_names_from_labels.json
    python scripts/apply_item_review.py data/raw/review/augment_names_from_labels.json --kind augments

In each labelled game, every own-board reading of bar position k (a template
ID) is paired with the label for slot k. A template whose pairings agree on
one augment (at least MIN_READS readings, AGREE of them the same) takes that
name; templates that end up with the same name are merged by the apply step.
Templates never read in a labelled game stay unnamed (label more games).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter, defaultdict

from src.utils import paths
from src.vision.arena import home_board_rows

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
OUT = paths.REPO / "data" / "raw" / "review" / "augment_names_from_labels.json"
MIN_READS = 3
AGREE = 0.8


def main():
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    if not labels:
        sys.exit("No augment labels yet - run scripts/label_augments.py")
    votes: dict[str, Counter] = defaultdict(Counter)
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
            for k, tid in enumerate(r.get("augments") or []):
                if tid and k < len(lab["augments"]) and lab["augments"][k] != "_unknown":
                    votes[tid][lab["augments"][k]] += 1
    names, unsure = {}, {}
    for tid, c in sorted(votes.items()):
        name, n = c.most_common(1)[0]
        total = sum(c.values())
        if total >= MIN_READS and n / total >= AGREE:
            names[tid] = name
        else:
            unsure[tid] = dict(c.most_common(3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"same": [], "different": [], "names": names}, indent=1), encoding="utf-8")
    print(f"{len(labels)} labelled games: {len(names)} templates named, {len(unsure)} unclear -> {OUT}")
    for tid, c in list(unsure.items())[:15]:
        print(f"  unclear {tid}: {c}")


if __name__ == "__main__":
    main()
