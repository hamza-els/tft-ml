"""
How well the pipeline names a game's augments, scored against Hamza's labels
(data/processed/augment_labels.json), one VOD held out at a time.

Usage:
    python scripts/eval_augments.py

For each VOD: templates are named from the labels of the OTHER VODs only (as
scripts/name_augment_templates.py does: every own-board reading of bar slot k
in a labelled game votes for that game's label k), then each held-out game's
per-game decision (src/state/sequence.game_augments, without its label) is
turned into names and compared slot by slot with the label. This is what a new,
unlabelled VOD gets.

Reports: slots right; slots whose augment was labelled in another VOD (the
only ones a footage template can name) and how many of those are right; and
the commonest confusions. "+" versions count as one augment
(GameData.augment_label); "_unknown" label slots are skipped.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections import Counter, defaultdict

from src.state.game_data import GameData
from src.state.sequence import game_augments
from src.utils import paths
from src.vision.arena import home_board_rows

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
MIN_READS, AGREE = 3, 0.8          # as in name_augment_templates.py


def load_game(key: str):
    vod, game = key.split("/")
    path = paths.STATES / vod / f"{game}.jsonl"
    if not path.exists():
        return None
    rows = sorted((json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l),
                  key=lambda r: r["frame"])
    rows = [r for r in rows if r.get("in_game")]
    return rows, home_board_rows(rows)


def main():
    # Score the template IDs themselves: game_augments would otherwise turn them into names
    # with today's lookup, which was built from every VOD's labels (the held-out VOD's too).
    import src.state.sequence as sequence
    sequence.augment_name = lambda template, tier: template
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", action="store_true",
                        help="Name a template per tier where its votes split between names")
    args = parser.parse_args()
    labels = {k: [GameData.augment_label(n) for n in v["augments"]]
              for k, v in json.loads(LABELS.read_text(encoding="utf-8")).items()}
    games = {k: g for k in labels if (g := load_game(k))}
    # per game: template-ID votes per slot (own-board readings), and the reader's decision
    votes_by_game, decided = {}, {}
    for key, (rows, home) in games.items():
        v = defaultdict(Counter)
        for r, h in zip(rows, home):
            if h:
                tiers = r.get("augment_tiers") or []
                for k, tid in enumerate(r.get("augments") or []):
                    if tid:
                        v[k][(tid, tiers[k] if k < len(tiers) else None)] += 1
        votes_by_game[key] = v
        names_, tiers_, _ = game_augments(rows, home)
        decided[key] = list(zip(names_, tiers_))

    total = right = possible = possible_right = 0
    confusions = Counter()
    per_vod = {}
    vods = sorted({k.split("/")[0] for k in games})
    for held in vods:
        # name templates from the other VODs' labels
        votes: dict[tuple, Counter] = defaultdict(Counter)
        seen_names = set()
        for key, lab in labels.items():
            if key.split("/")[0] == held or key not in votes_by_game:
                continue
            seen_names.update(lab)
            for k, c in votes_by_game[key].items():
                if k < len(lab) and lab[k] != "_unknown":
                    for tid, n in c.items():
                        votes[tid][lab[k]] += n
        by_tid: dict[str, Counter] = defaultdict(Counter)
        for (tid, tier), c in votes.items():
            by_tid[tid].update(c)

        def agreed(c: Counter):
            name, n = c.most_common(1)[0]
            return name if sum(c.values()) >= MIN_READS and n / sum(c.values()) >= AGREE else None

        names = {}
        for tid, c in by_tid.items():
            names[tid] = agreed(c)
        if args.tier:          # a template whose votes split: name it per tier
            for (tid, tier), c in votes.items():
                if names.get(tid) is None:
                    names[(tid, tier)] = agreed(c)

        def name_of(dec):
            tid, tier = dec
            return names.get(tid) or names.get((tid, tier))
        v_total = v_right = 0
        for key in games:
            if key.split("/")[0] != held:
                continue
            lab, dec = labels[key], decided[key]
            for k, true in enumerate(lab):
                if true.startswith("_"):
                    continue
                got = name_of(dec[k]) if k < len(dec) else None
                total += 1
                v_total += 1
                ok = got == true
                right += ok
                v_right += ok
                if true in seen_names:
                    possible += 1
                    possible_right += ok
                    if not ok:
                        confusions[(true, got or ("no decision" if k >= len(dec) else "unnamed template"))] += 1
        per_vod[held] = (v_right, v_total)

    print(f"{total} labelled slots in {len(games)} games, each VOD held out in turn:")
    print(f"  named right: {right}/{total} = {right / total:.0%}")
    print(f"  augment labelled in another VOD: {possible}/{total}; of those named right: "
          f"{possible_right}/{possible} = {possible_right / max(possible, 1):.0%}")
    for vod, (r, t) in per_vod.items():
        print(f"    {vod}: {r}/{t}")
    print("  commonest misses (label -> what the pipeline said), among augments seen in another VOD:")
    for (true, got), n in confusions.most_common(15):
        print(f"    {n}x  {true}  ->  {got}")


if __name__ == "__main__":
    main()
