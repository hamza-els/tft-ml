"""
Train an item classifier for the icons under units' health bars, and compare
it with the template matcher (src/vision/unit_bar.UnitItemReader).

Usage:
    python scripts/train_item_classifier.py               # compare on held-out games, then train the final model
    python scripts/train_item_classifier.py --no-final    # comparison only

Data: item crops (scripts/collect_item_crops.py; 33 px = 25 px icon + 4 px
around) and Hamza's labels (data/processed/item_crops/labels.json). Classes
are the item names plus "_none" (no item) and "_covered" (behind a tooltip);
"_unknown" crops are left out.

Pre-labels: crops nobody labelled, where the matcher is very sure (template
score >= PRE_SCORE), take the matcher's name. They add many examples of
common items; Hamza's labels add the hard cases.

Comparison: every labelled crop from the HOLDOUT game of each VOD is kept
out of training (pre-labels from those games too), and the three readers
are scored on it, per slot: the matcher (with the left-fill rule), a model
trained on Hamza's labels only, and one trained on labels + pre-labels.

Output: assets/item_classifier.pt (gitignored), with the class list.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import re
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.unit_bar import Bar, UnitItemReader, held_count
from src.vision.unit_classifier import build_model

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "assets" / "item_classifier.pt"
HOLDOUT = "game_02"
PRE_SCORE = 0.90
INPUT = 64
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(img, (INPUT, INPUT), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Shifts of a couple of px (the icon sits +-2 px off), brightness. No mirroring: icons have a handedness."""
    dx, dy = rng.randint(-2, 2), rng.randint(-1, 1)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10), 0, 255)
    return out.astype(np.uint8)


def game_of(name: str) -> str:
    return re.search(r"__(game_\d+)__", name).group(1)


def train(examples, classes, device, epochs=20, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    idx = {c: i for i, c in enumerate(classes)}
    model = build_model(len(classes)).to(device)
    counts = Counter(l for _, l in examples)
    weights = torch.tensor([1 / np.sqrt(counts.get(c, 1)) for c in classes], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights / weights.mean())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(examples) // 64 + 1))
    for epoch in range(epochs):
        model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        total = 0.0
        for b in range(0, len(order), 64):
            batch = [examples[i] for i in order[b:b + 64]]
            x = torch.stack([to_tensor(augment(img, rng)) for img, _ in batch]).to(device)
            y = torch.tensor([idx[l] for _, l in batch], device=device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(batch)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(examples):.3f}")
    return model


@torch.no_grad()
def predict(model, imgs, classes, device) -> list[str]:
    model.eval()
    out = []
    for b in range(0, len(imgs), 256):
        x = torch.stack([to_tensor(i) for i in imgs[b:b + 256]]).to(device)
        out += [classes[i] for i in model(x).argmax(1).cpu().tolist()]
    return out


def score(pred: list[str], truth: list[str]) -> dict:
    items = [(p, t) for p, t in zip(pred, truth) if not t.startswith("_")]
    empty = [(p, t) for p, t in zip(pred, truth) if t == "_none"]
    return {
        "items right": sum(p == t for p, t in items), "items": len(items),
        "items missed (said empty/covered)": sum(p.startswith("_") for p, _ in items),
        "items wrong name": sum((p != t and not p.startswith("_")) for p, t in items),
        "empty read as item": sum(not p.startswith("_") for p, _ in empty), "empty": len(empty),
    }


def matcher_predictions(names, index):
    """The matcher's decision per labelled slot: item name if the slot is within held_count, else _none."""
    cfg = vision_cfg["unit_bar"]
    reader = UnitItemReader(use_model=False)       # the matcher alone, as the baseline
    out, cache = [], {}
    for n in names:
        info = index[n]
        fp = info["frame_path"]
        if fp not in cache:
            cache = {fp: cv2.imread(fp)}
        frame = cache[fp]
        x, y = info["region"][:2]
        s = info["slot"]
        bar = Bar(left=x - cfg["item_dx"] - cfg["item_pitch"] * s, right=0, top=y - cfg["item_dy"])
        slots = reader.slots(frame, bar)
        name = slots[s]["name"] if s < held_count(slots) else "_none"
        out.append(name if name else "_unknown")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pre-labels", action="store_true",
                        help="Also train the final model on the matcher's confident pre-labels (didn't help on 2026-11-18)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = json.loads((paths.ITEM_DIR / "index.json").read_text(encoding="utf-8"))
    labels = json.loads((paths.ITEM_DIR / "labels.json").read_text(encoding="utf-8"))
    labels = {n: l for n, l in labels.items() if l != "_unknown" and (paths.ITEM_IMAGES / n).exists()}
    imgs = {n: cv2.imread(str(paths.ITEM_IMAGES / n)) for n in index if (paths.ITEM_IMAGES / n).exists()}
    pre = {n: info["guess"] for n, info in index.items()
           if n not in labels and n in imgs and info.get("guess") and not info["guess"].startswith("uitem_")
           and info.get("score", 0) >= PRE_SCORE}
    print(f"{len(labels)} labelled crops, {len(pre)} pre-labelled by the matcher (score >= {PRE_SCORE}), device {device}")

    test = sorted(n for n in labels if game_of(n) == HOLDOUT)
    train_lab = [(imgs[n], l) for n, l in labels.items() if game_of(n) != HOLDOUT]
    train_pre = [(imgs[n], l) for n, l in pre.items() if game_of(n) != HOLDOUT]
    truth = [labels[n] for n in test]
    print(f"Held out: {len(test)} labelled crops from every VOD's {HOLDOUT}")

    results = {"matcher": score(matcher_predictions(test, index), truth)}
    for name, data in (("model, labels only", train_lab), ("model, labels + pre-labels", train_lab + train_pre)):
        classes = sorted({l for _, l in data})
        print(f"  training {name}: {len(data)} examples, {len(classes)} classes")
        model = train(data, classes, device, args.epochs)
        results[name] = score(predict(model, [imgs[n] for n in test], classes, device), truth)
        unseen = sum(1 for t in truth if t not in classes)
        if unseen:
            print(f"    ({unseen} held-out crops are of items this model never saw)")

    print("\nOn held-out games:")
    for name, r in results.items():
        print(f"  {name:28s} items right {r['items right']}/{r['items']} ({r['items right'] / r['items']:.1%}), "
              f"missed {r['items missed (said empty/covered)']}, wrong {r['items wrong name']}; "
              f"empty read as item {r['empty read as item']}/{r['empty']}")

    if args.no_final:
        return
    data = [(imgs[n], l) for n, l in labels.items()]
    if args.pre_labels:
        data += [(imgs[n], l) for n, l in pre.items()]
    classes = sorted({l for _, l in data})
    print(f"\nFinal model on everything: {len(data)} examples, {len(classes)} classes")
    model = train(data, classes, device, args.epochs)
    torch.save({"classes": classes, "state_dict": model.state_dict(), "input": INPUT}, WEIGHTS)
    print(f"Saved {WEIGHTS}")


if __name__ == "__main__":
    main()
