"""
Train the planning-or-combat frame classifier (src/vision/frame_classifier.py).

Usage:
    python scripts/train_frame_classifier.py --collect     # sample frames -> ~/tft_data/frame_classifier/
    python scripts/train_frame_classifier.py --train       # train, test, save assets/frame_classifier.pt
    python scripts/train_frame_classifier.py --timing      # seconds it adds per video

Examples come from frames the timer is sure about, own board only:
  planning: planning by the timer sequence and by the frame check, timer 15-90% full
  combat:   not planning by either, after the round's timer refilled (timer above 10%)
Up to --per-game of each per game, spread over the game, from every VOD whose
full frames are on disk. Two VODs (--holdout) are kept out of training.

Tests: accuracy on the held-out VODs' timer labels; Hamza's "wrong frame"
labels (frames the rules took for planning; the model should say combat);
and ~2,000 frames his ordinary unit labels come from (planning; how many the
model would wrongly veto).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import re
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import paths
from src.vision.arena import home_board_rows, planning_rows
from src.vision.frame_classifier import INPUT_HW, WEIGHTS, FrameClassifier, build_model, shrink, to_tensor

DATA = paths.LOCAL / "frame_classifier"
CROPS = paths.REPO / "data" / "processed" / "unit_crops"


def sample_frames(per_game: int) -> list[tuple[str, int]]:
    """(frame path, label) with label 0 planning, 1 combat."""
    out = []
    rng = random.Random(0)
    for states in sorted(paths.STATES.glob("*/game_??.jsonl")):
        vod, game = states.parent.name, states.stem
        folder = paths.FRAMES / vod / game
        if not folder.exists():
            continue
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        plan, home = planning_rows(rows), home_board_rows(rows)
        planning, combat = [], []
        for r, p, h in zip(rows, plan, home):
            t = r.get("timer_fill")
            if not h or t is None:
                continue
            if p and r.get("planning") and 0.15 <= t <= 0.9:
                planning.append(r["frame"])
            elif not p and not r.get("planning") and t > 0.1:
                combat.append(r["frame"])
        for frames, label in ((planning, 0), (combat, 1)):
            for f in rng.sample(frames, min(per_game, len(frames))):
                out.append((str(folder / f"frame_{f:06d}.jpg"), label))
    return out


def collect(per_game: int):
    items = sample_frames(per_game)
    DATA.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(DATA / "X.npy", mode="w+", dtype=np.uint8, shape=(len(items), *INPUT_HW, 3))
    keep = []
    for i, (path, label) in enumerate(tqdm(items, desc="frames", unit="frame", mininterval=2)):
        img = cv2.imread(path)
        if img is None:
            continue
        X[len(keep)] = shrink(img)
        keep.append({"path": path, "label": label, "vod": Path(path).parents[1].name})
    X.flush()
    (DATA / "meta.json").write_text(json.dumps(keep), encoding="utf-8")
    print(f"{len(keep)} frames ({sum(k['label'] == 0 for k in keep)} planning, "
          f"{sum(k['label'] == 1 for k in keep)} combat) -> {DATA}")


def _batches(X, idx, labels, bs, train, rng):
    order = list(idx)
    if train:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        part = order[i:i + bs]
        imgs = []
        for j in part:
            im = X[j]
            if train:     # brightness jitter and a small shift: what varies between streams and arenas
                im = np.clip(im.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
                dx = rng.randint(-6, 6)
                im = np.roll(im, dx, axis=1)
            imgs.append(to_tensor(im))
        yield torch.stack(imgs), torch.tensor([labels[j] for j in part])


def train_model(X, idx, labels, epochs: int, device: str) -> nn.Module:
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    rng = random.Random(1)
    for ep in range(epochs):
        model.train()
        total = n = 0
        for xb, yb in _batches(X, idx, labels, 64, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(yb)
            n += len(yb)
        print(f"    epoch {ep + 1}/{epochs}  loss {total / n:.3f}", flush=True)
    return model.eval()


@torch.no_grad()
def predict(model, X, idx, device) -> np.ndarray:
    out = []
    for xb, _ in _batches(X, idx, [0] * len(X), 128, False, None):
        out += torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().tolist()
    return np.array(out)


@torch.no_grad()
def predict_paths(model, paths_list, device) -> np.ndarray:
    imgs = [cv2.imread(p) for p in paths_list]
    keep = [i for i, im in enumerate(imgs) if im is not None]
    out = np.full(len(paths_list), np.nan)
    for i in range(0, len(keep), 64):
        part = keep[i:i + 64]
        x = torch.stack([to_tensor(shrink(imgs[j])) for j in part]).to(device)
        out[part] = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
    return out


def label_frames() -> tuple[list[str], list[str]]:
    """Frames of Hamza's wrong-frame labels, and frames of his ordinary unit labels."""
    idx = json.loads((CROPS / "index.json").read_text(encoding="utf-8"))
    lab = json.loads((CROPS / "labels.json").read_text(encoding="utf-8"))
    wrong, ordinary = set(), set()
    for k, v in lab.items():
        if k not in idx or "__p" in k:
            continue
        if v == "_wrong_frame":
            wrong.add(idx[k]["frame_path"])
        elif not v.startswith("_"):
            ordinary.add(idx[k]["frame_path"])
    return sorted(wrong), sorted(ordinary - wrong)


def train(epochs: int, holdout: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(DATA / "X.npy", mmap_mode="r")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    labels = [m["label"] for m in meta]
    vods = sorted({m["vod"] for m in meta})
    held = holdout or vods[-2:]
    tr = [i for i, m in enumerate(meta) if m["vod"] not in held]
    te = [i for i, m in enumerate(meta) if m["vod"] in held]
    print(f"{len(meta)} frames; training on {len(tr)}, testing on {len(te)} from {held}; device {device}")
    model = train_model(X, tr, labels, epochs, device)
    p = predict(model, X, te, device)
    y = np.array([labels[i] for i in te])
    print(f"  held-out VODs (timer labels): {np.mean((p >= 0.5) == y):.1%} right "
          f"(planning kept {np.mean(p[y == 0] < 0.5):.1%}, combat caught {np.mean(p[y == 1] >= 0.5):.1%})")
    wrong, ordinary = label_frames()
    pw = predict_paths(model, wrong, device)
    po = predict_paths(model, ordinary, device)
    for t in (0.5, 0.7, 0.9):
        print(f"  veto at combat_p >= {t}: Hamza's wrong frames caught {np.nanmean(pw >= t):.0%} "
              f"({int(np.nansum(pw >= t))}/{int(np.sum(~np.isnan(pw)))}), his ordinary planning frames vetoed "
              f"{np.nanmean(po >= t):.1%} ({int(np.nansum(po >= t))}/{int(np.sum(~np.isnan(po)))})")
    (DATA / "test_scores.json").write_text(json.dumps({"wrong": dict(zip(wrong, pw.tolist())),
                                                      "ordinary": dict(zip(ordinary, po.tolist()))}), encoding="utf-8")
    print("  final model on every frame ...", flush=True)
    model = train_model(X, list(range(len(meta))), labels, epochs, device)
    # Saved beside the data, not over assets/frame_classifier.pt: a running batch starts using
    # the assets copy at its next VOD, so it is copied there by hand after the test numbers are read.
    torch.save({"state_dict": model.state_dict(), "input_hw": INPUT_HW}, DATA / "frame_classifier.pt")
    print(f"  saved {DATA / 'frame_classifier.pt'}  (copy to {WEIGHTS} to use it)")


def timing():
    """Seconds the veto adds per video: model time per planning frame x planning frames per video."""
    clf = FrameClassifier()
    counts = []
    for d in sorted(paths.STATES.iterdir()):
        n = 0
        for states in d.glob("game_??.jsonl"):
            rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                          key=lambda r: r["frame"])
            rows = [r for r in rows if r.get("in_game")]
            n += sum(1 for p, h in zip(planning_rows(rows), home_board_rows(rows)) if p and h)
        if n:
            counts.append(n)
    sample = sorted(paths.FRAMES.glob("*/game_*/frame_*.jpg"))[::997][:256]
    t0 = time.perf_counter()
    imgs = [cv2.imread(str(p)) for p in sample]
    t_read = (time.perf_counter() - t0) / len(sample)
    clf.combat_p(imgs[:8])                                  # warm-up
    t0 = time.perf_counter()
    clf.combat_p(imgs)
    t_model = (time.perf_counter() - t0) / len(sample)
    per_video = np.median(counts)
    print(f"own-board planning frames per video: median {per_video:.0f} (range {min(counts)}-{max(counts)})")
    print(f"per frame: reading the JPEG {t_read * 1000:.1f} ms, shrinking + model {t_model * 1000:.1f} ms")
    print(f"added per video: {per_video * t_model:.0f} s inside read_units (frames already read there), "
          f"{per_video * (t_model + t_read):.0f} s as a separate pass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--per-game", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--holdout", nargs="*", default=[])
    args = parser.parse_args()
    if args.collect:
        collect(args.per_game)
    if args.train:
        train(args.epochs, args.holdout)
    if args.timing:
        timing()


if __name__ == "__main__":
    main()
