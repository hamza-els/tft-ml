"""
Train the unit classifier on the crops labelled with scripts/label_units.py.

Usage:
    python scripts/train_unit_classifier.py
    python scripts/train_unit_classifier.py --holdout game_02 --epochs 40

Two passes:
  1. Train on every labelled game except --holdout and report accuracy on the
     held-out game (crops of a game the model never saw: the honest number).
  2. Train again on all labels and save assets/unit_classifier.pt, which
     label_units.py then uses for its suggestions.

Classes with very few labels are learnt poorly; the report lists accuracy per
class so you can see which champions need more examples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.utils import paths
from src.vision.unit_classifier import WEIGHTS, build_model, to_tensor

CROP_DIR = paths.CROP_DIR          # index.json, labels.json (repo)
IMAGES = paths.CROP_IMAGES         # crop images (local disk, src/utils/paths.py)


def load_examples() -> list[tuple[np.ndarray, str, str]]:
    """(crop, label, game) for every labelled crop still on disk."""
    labels = json.loads((CROP_DIR / "labels.json").read_text(encoding="utf-8"))
    out = []
    for name, label in sorted(labels.items()):
        if label in ("_wrong_frame", "_covered"):   # not classes: a wrong frame; a tooltip (now detected, tooltip.panel_in_crop)
            continue
        img = cv2.imread(str(IMAGES / name))
        m = re.search(r"__(game_\d+)__", name)
        if img is not None and m:
            out.append((img, label, m.group(1)))
    return out


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small shifts, scale, brightness and mirror: what varies between frames of the same unit."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.92, 1.08)
    dx, dy = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.05, 0.05) * h
    m = np.float32([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]])
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12), 0, 255)
    out = out.astype(np.uint8)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def train(examples, classes, epochs: int, device: str, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    index = {c: i for i, c in enumerate(classes)}
    model = build_model(len(classes)).to(device)

    # Rare classes get a larger weight so the common ones don't drown them out.
    counts = Counter(label for _, label, _ in examples)
    weights = torch.tensor([1.0 / np.sqrt(counts.get(c, 1)) for c in classes], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * ((len(examples) + 31) // 32))

    for epoch in range(epochs):
        model.train()
        order = list(range(len(examples)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), 32):
            batch = [examples[i] for i in order[start:start + 32]]
            x = torch.stack([to_tensor(augment(img, rng)) for img, _, _ in batch]).to(device)
            y = torch.tensor([index[label] for _, label, _ in batch], device=device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(examples):.3f}")
    return model.eval()


@torch.no_grad()
def evaluate(model, examples, classes, device: str):
    index = {c: i for i, c in enumerate(classes)}
    per_class = defaultdict(lambda: [0, 0])     # label -> [correct, total]
    top3 = 0
    for img, label, _ in examples:
        probs = torch.softmax(model(to_tensor(img).unsqueeze(0).to(device)), dim=1)[0]
        ranked = torch.argsort(probs, descending=True)[:3].tolist()
        per_class[label][1] += 1
        if label in index and ranked[0] == index[label]:
            per_class[label][0] += 1
        if label in index and index[label] in ranked:
            top3 += 1
    correct = sum(c for c, _ in per_class.values())
    return correct / len(examples), top3 / len(examples), per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="game_02", help="Game kept out of training for the accuracy check")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-final", action="store_true", help="Only run the held-out check, don't save a model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_examples()
    classes = sorted({label for _, label, _ in examples})
    print(f"{len(examples)} labelled crops, {len(classes)} classes, device {device}")

    train_set = [e for e in examples if e[2] != args.holdout]
    test_set = [e for e in examples if e[2] == args.holdout]
    if test_set and train_set:
        print(f"\n1. Accuracy check: train on {len(train_set)}, test on {len(test_set)} crops from {args.holdout}")
        model = train(train_set, classes, args.epochs, device)
        acc, top3, per_class = evaluate(model, test_set, classes, device)
        train_counts = Counter(label for _, label, _ in train_set)
        print(f"\n   Right first guess: {acc:.1%}   Right answer in top 3: {top3:.1%}")
        print("   Per class (held-out correct/total, training examples):")
        for label, (c, t) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
            print(f"     {label:<20} {c:>3}/{t:<3}  ({train_counts.get(label, 0)} in training)")
    else:
        print(f"Holdout {args.holdout} has no labels - skipping the accuracy check")

    if args.no_final:
        return
    print(f"\n2. Final model: training on all {len(examples)} crops")
    model = train(examples, classes, args.epochs, device)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": classes, "state_dict": model.state_dict(),
                "trained_on": len(examples)}, WEIGHTS)
    print(f"   Saved {WEIGHTS}")


if __name__ == "__main__":
    main()
