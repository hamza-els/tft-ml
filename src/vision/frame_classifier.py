"""
Planning or combat, from one frame of the player's own board.

The timer sequence (arena.planning_rows) is right on most frames, but now and
then a frame mid-fight passes every rule for an instant: no unit happens to
stand in the far band the per-frame check watches, and the combat timer also
drains (Hamza's "wrong frame" labels, 2026-11-20: 26 frames, all passing
every check). A small image model sees what the rules don't: units off their
hexes, spell effects, units facing the enemy. It is used as a veto: a frame
counts as planning only if the rules and the model agree (planning_rows reads
the row's "combat_p").

Input: the arena (FRAME_BOX), shrunk to INPUT_HW. Trained by
scripts/train_frame_classifier.py on frames the timer is sure about; weights
assets/frame_classifier.pt (gitignored).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = _ROOT / "assets" / "frame_classifier.pt"
FRAME_BOX = (280, 60, 1380, 820)      # x, y, w, h of the arena in a 1920x1080 frame (HUD columns left out)
INPUT_HW = (128, 216)                  # model input height, width
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def shrink(frame: np.ndarray) -> np.ndarray:
    """The arena of a frame at model size, BGR uint8."""
    x, y, w, h = FRAME_BOX
    return cv2.resize(frame[y:y + h, x:x + w], (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)


def to_tensor(small_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1))


def build_model(pretrained: bool = True) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)          # 0 = planning, 1 = combat
    return model


class FrameClassifier:
    def __init__(self, path: Path = WEIGHTS, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @staticmethod
    def available() -> bool:
        return WEIGHTS.exists()

    @torch.no_grad()
    def combat_p(self, frames: list[np.ndarray], batch: int = 64) -> list[float]:
        """Probability that each full frame shows a fight."""
        out = []
        for i in range(0, len(frames), batch):
            x = torch.stack([to_tensor(shrink(f)) for f in frames[i:i + batch]]).to(self.device)
            out += torch.softmax(self.model(x), dim=1)[:, 1].cpu().tolist()
        return out
