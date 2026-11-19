"""
Item classifier for the icons under units' health bars: a ResNet-18 trained
on Hamza's item labels (scripts/train_item_classifier.py; weights in
assets/item_classifier.pt, gitignored).

Used together with the template matcher (src/vision/unit_bar.UnitItemReader):
when the model is very sure (item_model_confidence) its answer is taken,
otherwise the matcher decides. On held-out games (2026-11-18): items the
model was trained on 309/310 right (matcher 293); items it never saw fall
back to the matcher (36/40); hybrid 345/350 vs matcher alone 329/350; no
empty slot read as an item. Every seen item scored >= 0.97, no unseen one
above 0.92.

Input: the 25 px icon with 4 px around it (33 px), as the item crops.
Classes: item names, "_none" (no item), "_covered" (behind a tooltip).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from src.vision.unit_classifier import build_model

_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = _ROOT / "assets" / "item_classifier.pt"
PAD = 4
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1))


class ItemClassifier:
    def __init__(self, path: Path = WEIGHTS, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.size = int(ckpt.get("input", 64))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @staticmethod
    def available() -> bool:
        return WEIGHTS.exists()

    @torch.no_grad()
    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """(class, probability) for each 33 px crop."""
        if not crops:
            return []
        x = torch.stack([to_tensor(c, self.size) for c in crops]).to(self.device)
        p = torch.softmax(self.model(x), 1).cpu().numpy()
        return [(self.classes[i], float(p[k, i])) for k, i in enumerate(p.argmax(1))]
