"""
Augment templates chosen from many named examples, instead of the first crop
of each icon the harvest happened to see.

Usage:
    python scripts/build_augment_templates.py --eval     # held-out test only (one VOD out at a time)
    python scripts/build_augment_templates.py --write    # write assets/templates/augments_v2/ + names

Names come from games whose augments are known: Hamza's label, else the stream
overlay (scripts/read_augment_overlay.py). The bar fills left to right in pick
order and the overlay lists them in the same order, so slot k of every
own-board frame of such a game shows its k-th augment. Per game up to
--per-game frames are sampled (fights excluded: both bars up), each icon cut
with AugmentReader at the counts the stage allows; crops unlike the rest of
their game slot are dropped (correlation < 0.6 with the slot's medoid).

Per augment ("+" versions folded): up to --k medoid templates, picked greedily
so every crop is close to one of them (the first is the crop most like all its
siblings; each next one helps the crops the others serve worst).

Same icon, different names: two augments whose templates correlate >= SAME at
the same colour tier are merged under the more common name (Commerce Core and
Shopping Spree share an icon, Hamza 2026-11-20); at different tiers they stay
apart and are told apart by tier (Silver / Gold Destiny). Both lists are
printed and, with --write, saved in assets/templates/augments_v2/groups.json.

Matching is as AugmentReader.match: template trimmed 2 px, best of 5x5
placements, normalised correlation, the best template over all wins. The
held-out test builds the templates without one VOD, names that VOD's labelled
games by per-slot vote and compares with Hamza's labels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from scripts.collect_augment_crops import keep_agreeing
from src.state.game_data import GameData
from src.utils import paths
from src.vision.arena import home_board_rows
from src.vision.augments import AugmentReader, expected_counts

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
OUT_DIR = paths.REPO / "assets" / "templates" / "augments_v2"
SAME = 0.90


def frame_dir(vod: str, game: str) -> Path | None:
    for root in (paths.FRAMES, paths.AUG_FRAMES):
        if (root / vod / game).exists():
            return root / vod / game
    return None


def known_names() -> dict[str, tuple[list[str], str]]:
    """game key -> (names in slot order, 'label' | 'overlay')."""
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    out = {k: (v["augments"], "label") for k, v in labels.items()}
    for ov_path in paths.STATES.glob("*/augment_overlay.json"):
        for game, v in json.loads(ov_path.read_text(encoding="utf-8")).items():
            key = f"{ov_path.parent.name}/{game}"
            if key not in out and v.get("augments"):
                out[key] = (v["augments"], "overlay")
    return out


def collect(per_game: int) -> list[dict]:
    """Crops of named games: {vod, game, slot, name, tier, icon}."""
    reader = AugmentReader()
    crops = []
    for key, (names, source) in tqdm(sorted(known_names().items()), desc="games", unit="game"):
        vod, game = key.split("/")
        states, folder = paths.STATES / vod / f"{game}.jsonl", frame_dir(vod, game)
        if not states.exists() or folder is None:
            continue
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        cands = [r for r, h in zip(rows, home_board_rows(rows)) if h and r.get("augments") and not r.get("away_augments")]
        if not cands:
            continue
        picks = [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, min(per_game, len(cands))).astype(int))]
        per_slot = defaultdict(list)
        for r in picks:
            img = cv2.imread(str(folder / f"frame_{r['frame']:06d}.jpg"))
            if img is None:
                continue
            for k, icon in enumerate(reader.icons(img, expected_counts(r.get("stage"), r.get("round_num")))):
                if k < len(names) and not names[k].startswith("_"):
                    per_slot[k].append((r["frame"], icon))
        for k, items in per_slot.items():
            for _, icon in keep_agreeing(items, 0.6):
                crops.append({"vod": vod, "game": game, "slot": k, "source": source,
                              "name": GameData.augment_label(names[k]), "tier": reader.tier(icon), "icon": icon})
    return crops


def _vec(icon: np.ndarray, trim: int = 0) -> np.ndarray:
    a = icon[trim:icon.shape[0] - trim, trim:icon.shape[1] - trim].astype(np.float32)
    a = a - a.reshape(-1, 3).mean(axis=0)
    n = float(np.linalg.norm(a))
    return (a / n if n else a).ravel()


def _windows(icon: np.ndarray) -> np.ndarray:
    """The 25 placements a 2-px-trimmed template can take inside the icon, as unit vectors."""
    h, w = icon.shape[:2]
    win = np.lib.stride_tricks.sliding_window_view(icon.astype(np.float32), (h - 4, w - 4, 3))[:, :, 0]
    p = win.reshape(-1, (h - 4) * (w - 4), 3)
    p = (p - p.mean(axis=1, keepdims=True)).reshape(len(p), -1)
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-6)


def medoids(icons: list[np.ndarray], k: int, sample: int = 150) -> list[int]:
    """Indices of up to k crops that together are close to all the others (greedy facility location)."""
    idx = list(range(len(icons)))
    if len(idx) > sample:
        idx = list(np.linspace(0, len(icons) - 1, sample).astype(int))
    V = np.stack([_vec(icons[i]) for i in idx])
    S = V @ V.T
    chosen, cover = [], np.full(len(idx), -1.0)
    for _ in range(min(k, len(idx))):
        gain = np.maximum(S, cover[None, :]).sum(axis=1) - cover.sum()
        gain[chosen] = -1
        j = int(gain.argmax())
        if chosen and gain[j] < 0.02 * len(idx):        # another template would barely help
            break
        chosen.append(j)
        cover = np.maximum(cover, S[j])
    return [idx[j] for j in chosen]


def build(crops: list[dict], k: int) -> tuple[list[np.ndarray], list[str], dict]:
    """Templates and their group names, plus {'merged': [...], 'by_tier': [...]}."""
    by_name = defaultdict(list)
    for c in crops:
        by_name[c["name"]].append(c)
    tmpl, tname = [], []
    for name, cs in sorted(by_name.items()):
        for i in medoids([c["icon"] for c in cs], k):
            tmpl.append(cs[i]["icon"])
            tname.append(name)
    # same icon under two names: best template-vs-template correlation per name pair
    tier_of = {n: Counter(c["tier"] for c in cs).most_common(1)[0][0] for n, cs in by_name.items()}
    T = np.stack([_vec(t, trim=2) for t in tmpl])
    W = [_windows(t) for t in tmpl]
    best = defaultdict(float)
    for a in range(len(tmpl)):
        sims = (W[a] @ T.T).max(axis=0)                 # template a's placements vs every template
        for b in range(len(tmpl)):
            if tname[a] != tname[b]:
                pair = tuple(sorted((tname[a], tname[b])))
                best[pair] = max(best[pair], float(sims[b]))
    merged, by_tier = [], []
    canon = {n: n for n in by_name}
    for (a, b), s in sorted(best.items(), key=lambda kv: -kv[1]):
        if s < SAME:
            continue
        if tier_of[a] == tier_of[b]:
            keep, drop = sorted((a, b), key=lambda n: (-len(by_name[n]), n))
            merged.append({"names": [keep, drop], "kept": keep, "similarity": round(s, 3), "tier": tier_of[a]})
        else:
            by_tier.append({"names": [a, b], "tiers": [tier_of[a], tier_of[b]], "similarity": round(s, 3)})
    for m in merged:                                    # follow chains (a->b->c) to one name
        for n in list(canon):
            if canon[n] == m["names"][1]:
                canon[n] = m["kept"]
    return tmpl, [canon[n] for n in tname], {"merged": merged, "by_tier": by_tier, "canonical": canon}


def tier_map(crops: list[dict], groups: dict) -> dict:
    """name -> {tier: name to use}: a name's own tiers, and for a same-icon pair at different tiers
    (Silver / Gold Destiny) the other name at the other tier."""
    canon = groups["canonical"]
    out = defaultdict(dict)
    for c in crops:
        out[canon.get(c["name"], c["name"])][c["tier"]] = canon.get(c["name"], c["name"])
    for m in groups["by_tier"]:
        (a, b), (ta, tb) = m["names"], m["tiers"]
        a, b = canon.get(a, a), canon.get(b, b)
        out[a][tb] = b
        out[b][ta] = a
    return out


def classify(crops: list[dict], tmpl: list[np.ndarray], tname: list[str]) -> list[str]:
    T = np.stack([_vec(t, trim=2) for t in tmpl])
    out = []
    for c in crops:
        sims = (_windows(c["icon"]) @ T.T).max(axis=0)
        out.append(tname[int(sims.argmax())])
    return out


def evaluate(crops: list[dict], k: int):
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    base = GameData.augment_base
    vods = sorted({c["vod"] for c in crops})
    slots = right = seen_slots = seen_right = 0
    misses = Counter()
    for held in tqdm(vods, desc="held-out", unit="vod"):
        train = [c for c in crops if c["vod"] != held]
        test = [c for c in crops if c["vod"] == held and f"{c['vod']}/{c['game']}" in labels]
        if not train or not test:
            continue
        tmpl, tname, groups = build(train, k)
        canon = groups["canonical"]
        guesses = classify(test, tmpl, tname)
        votes = defaultdict(Counter)
        for c, g in zip(test, guesses):
            votes[(c["game"], c["slot"])][(g, c["tier"])] += 1
        # per slot: the commonest name; where a name's icon is shared across tiers, the tier decides
        tier_names = tier_map(train, groups)
        train_names = {base(c["name"]) for c in train} | {base(v) for v in canon.values()}
        for (game, slot), v in votes.items():
            name_votes = Counter()
            for (g, tier), n in v.items():
                name_votes[g] += n
            g = name_votes.most_common(1)[0][0]
            tier = Counter({t: n for (gg, t), n in v.items() if gg == g}).most_common(1)[0][0]
            g = tier_names.get(g, {}).get(tier, g)
            truth = labels[f"{held}/{game}"]["augments"][slot]
            if truth.startswith("_"):
                continue
            t = GameData.augment_label(truth)
            ok = base(canon.get(g, g)) == base(canon.get(t, t))
            slots += 1
            right += ok
            if base(t) in train_names or base(canon.get(t, t)) in train_names:
                seen_slots += 1
                seen_right += ok
                if not ok:
                    misses[(t, g)] += 1
    print(f"\nHeld-out (one VOD out at a time), {slots} labelled slots with crops:")
    print(f"  right: {right}/{slots} = {right / max(slots, 1):.0%}")
    print(f"  augment named in another VOD (label or overlay): {seen_slots}; right {seen_right} = "
          f"{seen_right / max(seen_slots, 1):.0%}")
    for (t, g), n in misses.most_common(12):
        print(f"    {n}x {t} -> {g}")


def write(crops: list[dict], k: int):
    tmpl, tname, groups = build(crops, k)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    names, seen = {}, Counter()
    tier_names = tier_map(crops, groups)
    for icon, name in zip(tmpl, tname):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        stem = f"{slug}__{seen[slug]}"
        seen[slug] += 1
        cv2.imwrite(str(OUT_DIR / f"{stem}.png"), icon)
        names[stem] = name
        for tier, n in tier_names.get(name, {}).items():
            if n != name:
                names[f"{stem}@{tier}"] = n
    (OUT_DIR / "names.json").write_text(json.dumps(names, indent=1, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "groups.json").write_text(json.dumps({k: v for k, v in groups.items() if k != "canonical"}, indent=1),
                                        encoding="utf-8")
    print(f"{len(tmpl)} templates for {len(set(tname))} augments -> {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-game", type=int, default=40)
    parser.add_argument("--k", type=int, default=4, help="Templates per augment at most")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    crops = collect(args.per_game)
    by_source = Counter(c["source"] for c in crops)
    print(f"{len(crops)} crops of {len({c['name'] for c in crops})} augments "
          f"({by_source['label']} from labelled games, {by_source['overlay']} from overlay-named ones)")
    _, _, groups = build(crops, args.k)
    print("Same icon, same tier -> merged:")
    for m in groups["merged"]:
        print(f"  {m['names'][0]} + {m['names'][1]}  (kept {m['kept']}, correlation {m['similarity']}, {m['tier']})")
    print("Same icon, different tier -> told apart by tier:")
    for m in groups["by_tier"]:
        print(f"  {m['names'][0]} ({m['tiers'][0]}) / {m['names'][1]} ({m['tiers'][1]}), correlation {m['similarity']}")
    if args.eval:
        evaluate(crops, args.k)
    if args.write:
        write(crops, args.k)


if __name__ == "__main__":
    main()
