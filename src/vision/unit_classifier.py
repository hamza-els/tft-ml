"""
Champion classifier for unit crops (one board hex or bench slot window).

A small ImageNet-pretrained ResNet-18, fine-tuned on the crops labelled with
scripts/label_units.py. Class "_none" means the window holds no unit.
Trained by scripts/train_unit_classifier.py; weights in assets/unit_classifier.pt
(gitignored), with the class list stored alongside them.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = _ROOT / "assets" / "unit_classifier.pt"
INPUT_HW = (160, 128)           # crops are about 164x124 (board) and 140x116 (bench)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR crop -> normalised (3, H, W) float tensor at the model's input size."""
    h, w = INPUT_HW
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class UnitClassifier:
    def __init__(self, path: Path = WEIGHTS, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, top_k: int = 4) -> list[tuple[str, float]]:
        """Most likely classes with their probabilities, best first."""
        x = to_tensor(img_bgr).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.classes[i], float(probs[i])) for i in order]

    @staticmethod
    def available() -> bool:
        return WEIGHTS.exists()
