"""
Build the HUD text bank (src/vision/text_bank.py) from full-frame VODs.

Usage:
    python scripts/build_text_bank.py                       # all full-frame VODs -> assets/templates/text_bank/
    python scripts/build_text_bank.py --exclude knB6dE8Rbgg --out DIR   # for a held-out test
    python scripts/build_text_bank.py --crops-cache FILE    # reuse (or save) the collected crops

Per game, --per-game random frames; per field (stage, gold, level, streak, xp,
shop) the region is cut exactly as ocr_pipeline reads it, labelled with the raw
OCR value where it equals the cleaned value. Per value, up to K masks are kept
that together bring the most crops within the field's tolerance (greedy cover,
as the augment medoids); a mask within tolerance of another value's mask is
dropped (that picture would be ambiguous).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import paths
from src.vision.hud import HudDetector, shift
from src.vision.ocr_pipeline import REGIONS, SHOP_SLOTS, _OCR_CFG, crop_region
from src.vision.text_bank import BANK_DIR, INK, LEVEL

TOL = {"stage": 8, "gold": 8, "level": 8, "streak": 8, "xp": 8, "shop": 20}
K = {"shop": 20}          # masks per value (default 10)


def collect(per_game: int, exclude: set[str]) -> dict[str, list]:
    """field -> [(grey crop, value, vod)]"""
    hud_det = HudDetector()
    rng = random.Random(0)
    data = defaultdict(list)
    games = [s for s in sorted(paths.STATES.glob("*/game_??.jsonl"))
             if s.parent.name not in exclude and (paths.FRAMES / s.parent.name / s.stem).exists()
             and s.with_name(s.stem + ".clean.jsonl").exists()]
    for states in tqdm(games, desc="games", unit="game"):
        vod, game = states.parent.name, states.stem
        folder = paths.FRAMES / vod / game
        raw = {r["frame"]: r for r in (json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l)}
        clean = {r["frame"]: r for r in (json.loads(l) for l in
                 states.with_name(game + ".clean.jsonl").read_text(encoding="utf-8").splitlines() if l)}
        frames = [f for f in clean if f in raw]
        for f in rng.sample(frames, min(per_game, len(frames))):
            img = cv2.imread(str(folder / f"frame_{f:06d}.jpg"))
            if img is None:
                continue
            r, c = raw[f], clean[f]
            hud = hud_det.detect(img)

            def add(field, region, value):
                data[field].append((cv2.cvtColor(crop_region(img, region), cv2.COLOR_BGR2GRAY), value, vod))

            if hud.stage_visible and r.get("stage") and r.get("stage") == c.get("stage") \
                    and r.get("round_num") == c.get("round_num"):
                ix, iy = hud.stage_icon_xy
                dx, dy, w, h = _OCR_CFG["stage_text_offset"]
                add("stage", (ix + dx, iy + dy, w, h), f"{int(r['stage'])}-{int(r['round_num'])}")
            if not hud.shop_visible:
                continue
            for key, region in REGIONS.items():
                region = shift(region, hud.bottom_dx)
                if key == "xp":
                    if r.get("xp_current") is not None and r.get("xp_current") == c.get("xp_current") \
                            and r.get("xp_needed") == c.get("xp_needed"):
                        add("xp", region, f"{int(r['xp_current'])}/{int(r['xp_needed'])}")
                elif r.get(key) is not None and r.get(key) == c.get(key):
                    add(key, region, str(int(abs(r[key]))))
            s, cs = r.get("shop_slots") or [], c.get("shop_slots") or []
            for i, slot in enumerate(SHOP_SLOTS):
                if i < len(s) and i < len(cs) and s[i] == cs[i] and s[i]:
                    add("shop", shift(slot, hud.bottom_dx), s[i])
    return dict(data)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * (a @ b.T)


def medoids(M: np.ndarray, k: int, tol: int) -> list[int]:
    idx = np.arange(len(M))
    if len(M) > 400:
        idx = np.random.RandomState(0).choice(len(M), 400, replace=False)
    D = _dist(M[idx], M[idx])
    covered = np.zeros(len(idx), bool)
    out = []
    for _ in range(k):
        gain = ((D <= tol) & ~covered[None, :]).sum(1)
        j = int(gain.argmax())
        if gain[j] == 0:
            break
        out.append(int(idx[j]))
        covered |= D[j] <= tol
    return out


def build(data: dict[str, list], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for field, items in data.items():
        shape = Counter(g.shape for g, _, _ in items).most_common(1)[0][0]
        items = [(g, v) for g, v, _ in items if g.shape == shape and (g > LEVEL).sum() >= INK and v != "_empty"]
        M = np.stack([(g > LEVEL).ravel() for g, _ in items]).astype(np.float32)
        lab = np.array([v for _, v in items], dtype=object)
        tol, k = TOL[field], K.get(field, 10)
        bank, values = [], []
        for value in sorted(set(lab)):
            sel = M[lab == value]
            if len(sel) < 3:
                continue
            for j in medoids(sel, k, tol):
                bank.append(sel[j])
                values.append(value)
        B = np.stack(bank)
        D = _dist(B, B)
        vals = np.array(values, dtype=object)
        clash = ((D <= tol) & (vals[None, :] != vals[:, None])).any(1)
        B, vals = B[~clash], vals[~clash]
        np.savez_compressed(out_dir / f"{field}.npz", packed=np.packbits(B.astype(bool), axis=1),
                            values=np.array([str(v) for v in vals]), shape=np.array(shape), tol=np.array(tol))
        print(f"  {field:7s} {len(B):5d} masks for {len(set(vals)):3d} values "
              f"({int(clash.sum())} ambiguous dropped; from {len(items)} crops)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-game", type=int, default=150)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--out", default=str(BANK_DIR))
    parser.add_argument("--crops-cache", help="Pickle of collected crops: read if it exists, else written")
    args = parser.parse_args()
    cache = Path(args.crops_cache) if args.crops_cache else None
    if cache and cache.exists():
        data = pickle.loads(cache.read_bytes())
        data = {f: [it for it in items if it[2] not in set(args.exclude)] for f, items in data.items()}
    else:
        data = collect(args.per_game, set(args.exclude))
        if cache:
            cache.write_bytes(pickle.dumps(data))
    build(data, Path(args.out))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
