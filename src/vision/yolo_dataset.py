"""
Build a YOLO train/val dataset from hand-labeled frames.

Input (YOLO-format export from Roboflow / CVAT / LabelImg):
    data/processed/labeled/images/<vod_id>__...frame_XXXXXX.jpg
    data/processed/labeled/labels/<same stem>.txt
    data/processed/labeled/classes.txt    one champion name per line, in class-id order

Output (regenerated on every build):
    data/datasets/yolo/{train,val}/{images,labels}/
    data/datasets/yolo/data.yaml

The split is by VOD, not by frame: frames from one VOD look alike, so a random
frame split would leak near-duplicates into val and inflate the scores.
"""

import random
import shutil
from pathlib import Path

import yaml

from src.data_collection.label_sampler import SEP


def source_of(image_path: Path) -> str:
    """VOD id a labeled frame came from (prefix before the first separator)."""
    return image_path.stem.split(SEP)[0]


def game_of(image_path: Path) -> str:
    """VOD id + game folder (e.g. 'vod__game_03'); just the VOD id for unsplit frames."""
    parts = image_path.stem.split(SEP)
    return SEP.join(parts[:2]) if len(parts) > 2 else parts[0]


def load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def labeled_pairs(labeled_dir: Path) -> list[tuple[Path, Path]]:
    """(image, label) pairs. Images without a label file haven't been labeled yet and are skipped.
    An empty label file is kept — it marks a frame with no champions (a background example)."""
    pairs = []
    for img in sorted((labeled_dir / "images").glob("*.jpg")):
        lbl = labeled_dir / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def group_split(
    pairs: list[tuple[Path, Path]],
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    Split pairs into (train, val), holding out whole VODs for val.
    With a single VOD, whole games are held out instead.
    """
    rng = random.Random(seed)
    group_of = source_of
    groups = sorted({group_of(img) for img, _ in pairs})
    if len(groups) < 2:
        group_of = game_of
        groups = sorted({group_of(img) for img, _ in pairs})
        if len(groups) >= 2:
            print("  Only one VOD labeled — holding out whole games instead of whole VODs "
                  "(same streamer and HUD, so val scores will be somewhat optimistic).")

    if len(groups) < 2:
        print("  Only one game labeled — falling back to a random frame split "
              "(val scores will be optimistic).")
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(pairs) * val_frac))
        return shuffled[n_val:], shuffled[:n_val]

    rng.shuffle(groups)
    target = len(pairs) * val_frac
    val_groups: set[str] = set()
    n_val = 0
    # Always leave at least one group for training.
    for g in groups[:-1]:
        if n_val >= target:
            break
        val_groups.add(g)
        n_val += sum(1 for img, _ in pairs if group_of(img) == g)

    train = [p for p in pairs if group_of(p[0]) not in val_groups]
    val = [p for p in pairs if group_of(p[0]) in val_groups]
    return train, val


def build_dataset(
    labeled_dir: str = "data/processed/labeled",
    out_dir: str = "data/datasets/yolo",
    val_frac: float = 0.15,
    seed: int = 0,
) -> Path:
    """Copy labeled frames into a fresh YOLO dataset and return the data.yaml path."""
    labeled = Path(labeled_dir)
    out = Path(out_dir).resolve()

    classes_file = labeled / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"{classes_file} not found — export classes.txt with your labels")
    classes = load_classes(classes_file)

    pairs = labeled_pairs(labeled)
    if not pairs:
        raise RuntimeError(f"No labeled images in {labeled}/images with matching labels/")

    train, val = group_split(pairs, val_frac, seed)

    for split, items in (("train", train), ("val", val)):
        split_dir = out / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir(parents=True)
        for img, lbl in items:
            shutil.copy2(img, split_dir / "images" / img.name)
            shutil.copy2(lbl, split_dir / "labels" / lbl.name)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(classes)),
    }, sort_keys=False))

    print(f"  {len(classes)} classes | train {len(train)} | val {len(val)} "
          f"({len({game_of(i) for i, _ in val})} held-out game(s))")
    return data_yaml
