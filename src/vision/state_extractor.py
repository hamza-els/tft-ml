"""
Combines OCR + template matching + champion detection into a single StateVector
per frame. This is the output that feeds the Macro Engine.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from src.vision.champion_detector import ChampionDetector, Detection
from src.vision.ocr_pipeline import extract_ui_values
from src.vision.template_matcher import load_templates, match_template


@dataclass
class UnitState:
    name: str
    star: int
    items: list[str]
    region: str  # "board" | "bench" | "shop"
    hex_position: tuple[int, int] | None = None  # (col, row) on board grid


@dataclass
class StateVector:
    # --- Scalars from OCR ---
    stage: float | None = None
    round_num: float | None = None
    gold: float | None = None
    hp: float | None = None
    level: float | None = None

    # --- Units ---
    board: list[UnitState] = field(default_factory=list)
    bench: list[UnitState] = field(default_factory=list)
    shop: list[UnitState] = field(default_factory=list)

    # --- Items on bench (unequipped) ---
    bench_items: list[str] = field(default_factory=list)

    # --- Augments (detected via template matching) ---
    augments: list[str] = field(default_factory=list)

    # --- Opponent context (populated by OpponentMemory) ---
    opponent_id: str | None = None
    scouted_board: list[UnitState] = field(default_factory=list)
    scout_staleness: int = 0  # turns since last scout

    def to_dict(self) -> dict:
        return asdict(self)


class StateExtractor:
    def __init__(
        self,
        detector: ChampionDetector,
        item_template_dir: str,
        augment_template_dir: str,
    ):
        self.detector = detector
        self.item_templates = load_templates(item_template_dir)
        self.augment_templates = load_templates(augment_template_dir)

    def extract(self, frame: np.ndarray) -> StateVector:
        sv = StateVector()

        # 1. OCR for scalar UI values
        ui = extract_ui_values(frame)
        sv.gold = ui.get("gold")
        sv.hp = ui.get("hp")
        sv.level = ui.get("level")
        sv.stage = ui.get("stage")

        # 2. Champion detection
        detections: list[Detection] = self.detector.detect(frame)
        for d in detections:
            unit = UnitState(name=d.label, star=d.star_level, items=[], region=d.region)
            if d.region == "board":
                sv.board.append(unit)
            elif d.region == "bench":
                sv.bench.append(unit)
            else:
                sv.shop.append(unit)

        # 3. Template matching for augments
        for name, tmpl in self.augment_templates.items():
            hits = match_template(frame, tmpl, threshold=0.82)
            if hits:
                sv.augments.append(name)

        return sv
