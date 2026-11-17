"""
Train the champion detector on hand-labeled frames and install the weights.

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 150 --imgsz 1280 --batch 8
    python scripts/train_yolo.py --model assets/yolo_tft.pt   # fine-tune previous round

Steps:
  1. Rebuild data/datasets/yolo/ from data/processed/labeled/ (split by VOD).
  2. Train with Ultralytics; runs are saved under data/datasets/yolo/runs/.
  3. Copy best.pt to the weights path in configs/vision_config.yaml.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil

import yaml

from src.vision.yolo_dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default="data/processed/labeled")
    parser.add_argument("--dataset", default="data/datasets/yolo")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Starting weights (default: yolov8n.pt, COCO-pretrained)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Train resolution — keep high, units are small at 1080p")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    parser.add_argument("--no-install", action="store_true",
                        help="Don't copy best.pt over the configured weights")
    args = parser.parse_args()

    print("Building dataset...")
    data_yaml = build_dataset(args.labeled, args.dataset, args.val_frac)

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str((Path(args.dataset) / "runs").resolve()),
        name="champions",
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    if not args.no_install:
        with open("configs/vision_config.yaml") as f:
            weights = Path(yaml.safe_load(f)["champion_detector"]["weights"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, weights)
        print(f"Installed → {weights}")


if __name__ == "__main__":
    main()
