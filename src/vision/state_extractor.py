"""
Combines OCR + template matching + champion detection into a single StateVector
per frame. This is the output that feeds the Macro Engine.

The champion detector is optional: until YOLO weights are trained, board and
bench stay empty and everything else (OCR scalars, HP, shop) is still filled.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.utils.config import vision_cfg
from src.vision.arena import arena_signature, is_planning, timer_fill
from src.vision.augments import AugmentReader
from src.vision.champion_detector import ChampionDetector, Detection
from src.vision.hud import HudDetector
from src.vision.items import ItemColumnReader
from src.vision.ocr_pipeline import extract_ui_and_lobby, read_shop

_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class UnitState:
    name: str
    star: int | None   # None until a star-level reader exists; shop units are always 1
    items: list[str]
    region: str  # "board" | "bench" | "shop"
    hex_position: tuple[int, int] | None = None  # (col, row) on board grid


@dataclass
class StateVector:
    # --- HUD presence ---
    in_game: bool = False        # stage bar on screen
    shop_visible: bool = False   # bottom bar on screen (False in carousel / shop collapsed)
    planning: bool = False       # no health bars on the far part of the board
    timer_fill: float | None = None   # round timer bar, 0-1; refills when combat starts
    # Colour signature of the arena on screen. Whether that is the player's
    # own board is decided per game in src/state/sequence.py (home_board).
    arena_sig: list[float] = field(default_factory=list)

    # --- Scalars from OCR ---
    stage: float | None = None
    round_num: float | None = None
    gold: float | None = None
    hp: float | None = None
    level: float | None = None
    xp_current: float | None = None
    xp_needed: float | None = None
    streak: float | None = None   # signed: + win streak, - loss streak

    # --- Lobby ---
    other_hp: list[float] = field(default_factory=list)  # other players' HP, top to bottom

    # --- Units ---
    board: list[UnitState] = field(default_factory=list)
    bench: list[UnitState] = field(default_factory=list)
    shop: list[UnitState] = field(default_factory=list)
    shop_slots: list[str | None] = field(default_factory=list)  # per slot, None = empty

    # --- Unequipped items (left-hand item column), top to bottom ---
    bench_items: list[str | None] = field(default_factory=list)   # None = icon with no template yet

    # --- Augments picked so far (icon bar above the board), left to right ---
    augments: list[str | None] = field(default_factory=list)   # None = icon with no template yet
    augment_tiers: list[str] = field(default_factory=list)     # "silver" | "gold" | "prismatic"
    # Icons in the bar on the right: another player's augments, shown while
    # scouting or fighting on their board. Above 0 means not the player's own board.
    away_augments: int = 0

    # --- Opponent context (populated by OpponentMemory) ---
    opponent_id: str | None = None
    scouted_board: list[UnitState] = field(default_factory=list)
    scout_staleness: int = 0  # turns since last scout

    def to_dict(self) -> dict:
        return asdict(self)


class StateExtractor:
    def __init__(
        self,
        detector: ChampionDetector | None = None,
        hud: HudDetector | None = None,
        augments: AugmentReader | None = None,
        items: ItemColumnReader | None = None,
    ):
        self.detector = detector
        self.hud = hud or HudDetector()
        self.augments = augments or AugmentReader()
        self.items = items or ItemColumnReader()

    @classmethod
    def from_config(cls) -> "StateExtractor":
        """Build from vision_config.yaml; the detector is skipped if its weights don't exist yet."""
        det_cfg = vision_cfg["champion_detector"]
        weights = _ROOT / det_cfg["weights"]
        detector = None
        if weights.exists():
            detector = ChampionDetector(str(weights), det_cfg["board_max_ratio"],
                                        det_cfg["bench_max_ratio"])
        return cls(detector)

    def extract(self, frame: np.ndarray) -> StateVector:
        sv = StateVector()
        hud = self.hud.detect(frame)
        sv.in_game = hud.stage_visible
        sv.shop_visible = hud.shop_visible
        if not hud.stage_visible:
            return sv
        sv.arena_sig = arena_signature(frame)
        sv.planning = is_planning(frame)
        sv.timer_fill = timer_fill(frame)

        # 1. OCR for scalar UI values
        ui, sv.other_hp = extract_ui_and_lobby(frame, hud)
        sv.gold = ui["gold"]
        sv.hp = ui["hp"]
        sv.level = ui["level"]
        sv.stage = ui["stage"]
        sv.round_num = ui["round"]
        sv.xp_current = ui["xp_current"]
        sv.xp_needed = ui["xp_needed"]
        sv.streak = ui["streak"]

        # 2. Champion detection (board + bench)
        if self.detector is not None:
            detections: list[Detection] = self.detector.detect(frame)
            for d in detections:
                unit = UnitState(name=d.label, star=None, items=[], region=d.region)
                if d.region == "board":
                    sv.board.append(unit)
                else:
                    sv.bench.append(unit)

        # 3. Shop via OCR of card names
        if hud.shop_visible:
            sv.shop_slots = read_shop(frame, hud)
            sv.shop = [UnitState(name=name, star=1, items=[], region="shop")
                       for name in sv.shop_slots if name is not None]

        # 4. Unequipped items; look-alike templates of one item ("Name~2") share its name
        sv.bench_items = [n.split("~")[0] if n else None for n in self.items.read(frame)]

        # 5. Augment bar
        for aug in self.augments.read(frame):
            sv.augments.append(aug.name)
            sv.augment_tiers.append(aug.tier)
        sv.away_augments = self.augments.count_away(frame)

        return sv
