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
their game slot are dropped (correlation < 0.6 with the slot's medoid). In labelled games
a crop must instead match the icon the label was made on (ANCHOR; ~/tft_data/augment_crops/),
so frames of another player's bar (after his elimination) or a miscounted bar are left out.

Per augment ("+" versions folded): up to --k medoid templates, picked greedily
so every crop is close to one of them (the first is the crop most like all its
siblings; each next one helps the crops the others serve worst).

Game slots whose icon looks more like another augment's typical icon than its own
name's (by ODD_MARGIN) are left out and listed: label slips or overlay mix-ups.

Same icon, different names: two augments whose typical icons correlate >= SAME at
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
from src.state.sequence import own_span
from src.utils import paths
from src.vision.arena import home_board_rows
from src.vision.augments import AugmentReader, expected_counts
from src.vision.augment_art import render as render_art

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
OUT_DIR = paths.REPO / "assets" / "templates" / "augments_v2"
OFFICIAL = paths.REPO / "assets" / "templates" / "augments_official"
ART = "art__"        # stem prefix of templates rendered from official art (src/vision/augment_art.py)
SAME = 0.90
LABELLED_ICONS = paths.LOCAL / "augment_crops"   # the icon each label was made on (label_augments.py)
ANCHOR = 0.7        # a labelled game's crop must match its labelled icon this well
ODD_MARGIN = 0.05   # a game slot whose icon matches another name's typical icon this much better is left out
ODD_OTHER = 0.8     # ... and that well (Booster Pack's icons match nothing well: 0.5-0.6 to anything)
# Same icon by Hamza's word (2026-11-20), merged even where no game shows both: (kept, merged in)
MANUAL_MERGES = [("Commerce Core", "Shopping Spree")]


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
        if not states.exists() or folder is None or states.with_name(f"{game}.excluded.json").exists():
            continue                            # (excluded: another game mode, clean_states.py)
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        start, end, _ = own_span(rows)          # not another player's HUD after he is out
        rows = rows[start:end]
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
            kept = keep_agreeing(items, 0.6)
            # labelled games: only crops like the icon Hamza labelled (label_augments.py saves it), so
            # frames of another player's bar (spectating after elimination) or a miscounted bar don't count
            anchor = cv2.imread(str(LABELLED_ICONS / f"{vod}__{game}__a{k}.png")) if source == "label" else None
            if anchor is not None:
                if anchor.shape != items[0][1].shape:
                    anchor = cv2.resize(anchor, items[0][1].shape[1::-1])
                kept = [(f, icon) for f, icon in items if _sim(icon, anchor) >= ANCHOR]
            for _, icon in kept:
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


def _sim(a: np.ndarray, b: np.ndarray) -> float:
    """Match score of icon b as a template against icon a (as AugmentReader.match)."""
    return float((_windows(a) @ _vec(b, trim=2)).max())


def typical_icons(crops: list[dict]) -> tuple[dict, dict]:
    """
    Per name, its typical icon, and per game slot its own icon (the slot's medoid crop).
    Typical = the game slot most like the name's other game slots (with one or two games:
    the medoid of all its crops).
    """
    slots = defaultdict(list)
    for c in crops:
        slots[(c["name"], c["vod"], c["game"], c["slot"])].append(c["icon"])
    slot_icon = {key: icons[medoids(icons, 1)[0]] for key, icons in slots.items()}
    per_name = defaultdict(list)
    for key in slot_icon:
        per_name[key[0]].append(key)
    typical = {}
    for name, keys in per_name.items():
        if len(keys) >= 3:
            S = np.array([[_sim(slot_icon[a], slot_icon[b]) for b in keys] for a in keys])
            typical[name] = slot_icon[keys[int(S.sum(axis=1).argmax())]]
        else:
            icons = [c["icon"] for c in crops if c["name"] == name]
            typical[name] = icons[medoids(icons, 1)[0]]
    return typical, slot_icon


def odd_slots(crops: list[dict], margin: float = ODD_MARGIN) -> list[dict]:
    """
    Game slots whose icon looks more like another augment's typical icon than like its own
    name's (by `margin`): almost always a label slip or an overlay mix-up. Their crops are
    left out of the templates (one such game used to become a template of the wrong icon and
    then merge two names: Pandora's Items II took Solo Plate's scroll, 2026-11-23).
    """
    typical, slot_icon = typical_icons(crops)
    names = sorted(typical)
    T = np.stack([_vec(typical[n], trim=2) for n in names])
    out = []
    for (name, vod, game, slot), icon in slot_icon.items():
        sims = (_windows(icon) @ T.T).max(axis=0)
        own = float(sims[names.index(name)])
        j = int(sims.argmax())
        if names[j] != name and sims[j] - own > margin and sims[j] >= ODD_OTHER:
            out.append({"vod": vod, "game": game, "slot": slot, "name": name,
                        "looks_like": names[j], "own": round(own, 3), "other": round(float(sims[j]), 3)})
    return out


def build(crops: list[dict], k: int) -> tuple[list[np.ndarray], list[str], dict]:
    """Templates and their group names, plus {'merged': [...], 'by_tier': [...], 'odd': [...]}."""
    odd = odd_slots(crops)
    skip = {(o["vod"], o["game"], o["slot"]) for o in odd}
    crops = [c for c in crops if (c["vod"], c["game"], c["slot"]) not in skip]
    by_name = defaultdict(list)
    for c in crops:
        by_name[c["name"]].append(c)
    tmpl, tname = [], []
    for name, cs in sorted(by_name.items()):
        for i in medoids([c["icon"] for c in cs], k):
            tmpl.append(cs[i]["icon"])
            tname.append(name)
    # same icon under two names: their typical icons match (not any template of each: one stray
    # game's template used to be enough to merge two different icons)
    tier_of = {n: Counter(c["tier"] for c in cs).most_common(1)[0][0] for n, cs in by_name.items()}
    typical, _ = typical_icons(crops)
    names = sorted(typical)
    T = np.stack([_vec(typical[n], trim=2) for n in names])
    best = {}
    for a, na in enumerate(names):
        sims = (_windows(typical[na]) @ T.T).max(axis=0)
        for b, nb in enumerate(names):
            if na != nb:
                pair = tuple(sorted((na, nb)))
                best[pair] = max(best.get(pair, 0.0), float(sims[b]))
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
    for keep, drop in MANUAL_MERGES:
        if not any(set(m["names"]) == {keep, drop} for m in merged):
            merged.append({"names": [keep, drop], "kept": keep, "similarity": None, "tier": None, "manual": True})
        canon.setdefault(keep, keep)
        canon.setdefault(drop, drop)
    for m in merged:                                    # follow chains (a->b->c) to one name
        for n in list(canon):
            if canon[n] == m["names"][1]:
                canon[n] = m["kept"]
    return tmpl, [canon[n] for n in tname], {"merged": merged, "by_tier": by_tier, "canonical": canon, "odd": odd}


def art_templates(have: set[str], canon: dict) -> tuple[list[np.ndarray], list[str]]:
    """Official art rendered as bar icons, for augments with no footage template ('+' folded)."""
    have_base = {GameData.augment_base(canon.get(n, n)) for n in have}
    tmpl, tname = [], []
    for path in sorted(OFFICIAL.glob("*.png")):
        name = GameData.augment_label(path.stem.rpartition("__")[0])
        name = canon.get(name, name)
        if GameData.augment_base(name) in have_base:
            continue
        art = cv2.imread(str(path))
        if art is not None:
            tmpl.append(render_art(art))
            tname.append(name)
    return tmpl, tname


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


def classify(crops: list[dict], tmpl: list[np.ndarray], tname: list[str], is_art: list[bool] | None = None,
             bonus: float = 0.0) -> list[str]:
    """Best template's name per crop; art templates' scores get `bonus` (they score lower than footage)."""
    T = np.stack([_vec(t, trim=2) for t in tmpl])
    add = bonus * np.asarray(is_art if is_art is not None else [False] * len(tmpl), np.float32)
    out = []
    for c in crops:
        sims = (_windows(c["icon"]) @ T.T).max(axis=0) + add
        out.append(tname[int(sims.argmax())])
    return out


def evaluate(crops: list[dict], k: int, art: bool = False, bonus: float = 0.0):
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
        is_art = [False] * len(tmpl)
        if art:
            at, an = art_templates(set(tname), canon)
            tmpl, tname, is_art = tmpl + at, tname + an, is_art + [True] * len(at)
        guesses = classify(test, tmpl, tname, is_art, bonus)
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
    unseen = slots - seen_slots
    print(f"\nHeld-out (one VOD out at a time){', with official-art templates, bonus ' + str(bonus) if art else ''},"
          f" {slots} labelled slots with crops:")
    print(f"  right: {right}/{slots} = {right / max(slots, 1):.0%}")
    print(f"  augment named in another VOD (label or overlay): {seen_slots}; right {seen_right} = "
          f"{seen_right / max(seen_slots, 1):.0%}")
    print(f"  augment never seen in another VOD: {unseen}; right {right - seen_right} = "
          f"{(right - seen_right) / max(unseen, 1):.0%}")
    for (t, g), n in misses.most_common(12):
        print(f"    {n}x {t} -> {g}")


def write(crops: list[dict], k: int, art: bool = True):
    tmpl, tname, groups = build(crops, k)
    n_footage = len(tmpl)
    if art:
        at, an = art_templates(set(tname), groups["canonical"])
        tmpl, tname = tmpl + at, tname + an
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    names, seen = {}, Counter()
    tier_names = tier_map(crops, groups)
    for i, (icon, name) in enumerate(zip(tmpl, tname)):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        stem = f"{ART if i >= n_footage else ''}{slug}__{seen[slug]}"
        seen[slug] += 1
        cv2.imwrite(str(OUT_DIR / f"{stem}.png"), icon)
        names[stem] = name
        for tier, n in tier_names.get(name, {}).items():
            if n != name:
                names[f"{stem}@{tier}"] = n
    (OUT_DIR / "names.json").write_text(json.dumps(names, indent=1, sort_keys=True), encoding="utf-8")
    aliases = {n: c for n, c in groups["canonical"].items() if n != c}
    (OUT_DIR / "aliases.json").write_text(json.dumps(aliases, indent=1, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "groups.json").write_text(json.dumps({k: v for k, v in groups.items() if k != "canonical"}, indent=1),
                                        encoding="utf-8")
    print(f"{n_footage} footage templates for {len(set(tname[:n_footage]))} augments, "
          f"{len(tmpl) - n_footage} rendered from official art for {len(set(tname[n_footage:]))} more -> {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-game", type=int, default=40)
    parser.add_argument("--k", type=int, default=4, help="Templates per augment at most")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-art", action="store_true", help="No templates rendered from official art")
    parser.add_argument("--art-bonus", type=float, nargs="*", default=None,
                        help="--eval: also test with art templates, art scores + each bonus")
    args = parser.parse_args()
    crops = collect(args.per_game)
    by_source = Counter(c["source"] for c in crops)
    print(f"{len(crops)} crops of {len({c['name'] for c in crops})} augments "
          f"({by_source['label']} from labelled games, {by_source['overlay']} from overlay-named ones)")
    _, _, groups = build(crops, args.k)
    print("Same icon, same tier -> merged:")
    for m in groups["merged"]:
        print(f"  {m['names'][0]} + {m['names'][1]}  (kept {m['kept']}, correlation {m['similarity']}, {m['tier']})")
    print(f"Game slots left out (icon looks like another augment; label slip?): {len(groups['odd'])}")
    for o in groups["odd"]:
        src = "label" if f"{o['vod']}/{o['game']}" in json.loads(LABELS.read_text(encoding="utf-8")) else "overlay"
        print(f"  {o['vod']}/{o['game']} slot {o['slot'] + 1}: {src} {o['name']} ({o['own']}), "
              f"looks like {o['looks_like']} ({o['other']})")
    print("Same icon, different tier -> told apart by tier:")
    for m in groups["by_tier"]:
        print(f"  {m['names'][0]} ({m['tiers'][0]}) / {m['names'][1]} ({m['tiers'][1]}), correlation {m['similarity']}")
    if args.eval:
        evaluate(crops, args.k)
        for b in args.art_bonus or []:
            evaluate(crops, args.k, art=True, bonus=b)
    if args.write:
        write(crops, args.k, art=not args.no_art)


if __name__ == "__main__":
    main()
