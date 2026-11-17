"""
Detects whether the TFT HUD is on screen, using two small icon templates.

  stage icon — left end of the top stage bar. Visible for the whole game,
               including carousel and while spectating after elimination.
               It sits further right during stage 1 (shorter round bar), so
               it is searched for across the bar rather than at a fixed x.
  gold icon  — the coin in the bottom bar. Visible only while the player is
               alive and the shop bar is shown (hidden during carousel).

Templates and search windows are specific to one HUD layout (UI scale +
stream layout) and are set in configs/vision_config.yaml under `hud`.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg

_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HudState:
    stage_visible: bool
    shop_visible: bool
    stage_icon_xy: tuple[int, int] | None   # top-left of the stage icon match
    stage_one_layout: bool                  # True while the bar is in its stage-1 position


class HudDetector:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or vision_cfg["hud"]
        tdir = _ROOT / cfg["template_dir"]
        self.stage_icon = self._load(tdir / "stage_icon.png")
        self.gold_icon = self._load(tdir / "gold_icon.png")
        self.stage_search = tuple(cfg["stage_icon_search"])
        self.gold_search = tuple(cfg["gold_icon_search"])
        self.threshold = float(cfg["match_threshold"])
        self.stage_one_min_x = int(cfg["stage_one_min_x"])

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"HUD template not found: {path}")
        return img

    @staticmethod
    def _match(frame: np.ndarray, template: np.ndarray,
               window: tuple[int, int, int, int]) -> tuple[float, int, int]:
        x, y, w, h = window
        res = cv2.matchTemplate(frame[y:y + h, x:x + w], template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return float(score), x + loc[0], y + loc[1]

    def detect(self, frame: np.ndarray) -> HudState:
        s_score, sx, sy = self._match(frame, self.stage_icon, self.stage_search)
        g_score, _, _ = self._match(frame, self.gold_icon, self.gold_search)
        stage_visible = s_score >= self.threshold
        return HudState(
            stage_visible=stage_visible,
            shop_visible=g_score >= self.threshold,
            stage_icon_xy=(sx, sy) if stage_visible else None,
            stage_one_layout=stage_visible and sx >= self.stage_one_min_x,
        )
