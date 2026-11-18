"""
OCR pipeline for extracting UI values from TFT frames.

Reads gold, level, XP, streak and stage from fixed pixel regions, player HP
from the right-hand player list, and champion names from the five shop cards.

Regions are loaded from configs/vision_config.yaml (all assume 1920x1080) and
are specific to one HUD layout — see the note at the top of that file.
"""

import difflib
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch

from src.utils.config import vision_cfg
from src.vision.hud import HudDetector, HudState, shift

_ROOT = Path(__file__).resolve().parents[2]

# One OCR text detection: (x, y, w, h) in frame pixels, text, confidence.
TextBox = tuple[tuple[int, int, int, int], str, float]


def _load_champion_list(path: str | None) -> list[str]:
    if not path or not (_ROOT / path).exists():
        return []
    lines = (_ROOT / path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


_OCR_CFG = vision_cfg["ocr"]
REGIONS: dict[str, tuple[int, int, int, int]] = {
    k: tuple(v) for k, v in _OCR_CFG["regions"].items()
}
SHOP_SLOTS: list[tuple[int, int, int, int]] = [tuple(v) for v in _OCR_CFG["shop_slots"]]
CHAMPIONS: list[str] = _load_champion_list(_OCR_CFG.get("champion_list"))

# Per-region OCR allowlists
_ALLOWLIST: dict[str, str] = {
    "stage":  "0123456789-",   # reads "3-2"
    "xp":     "0123456789/",   # reads "2/6"
    "shop":   "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '.&",  # "Cho'Gath", "Dr. Mundo"
    "default":"0123456789",
}

_reader: easyocr.Reader | None = None
_hud: HudDetector | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _reader


def _get_hud() -> HudDetector:
    global _hud
    if _hud is None:
        _hud = HudDetector()
    return _hud


def crop_region(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = region
    return frame[max(0, y):y + h, max(0, x):x + w]


def _read_boxes(frame: np.ndarray, region: tuple[int, int, int, int],
                allowlist: str, scale: float) -> list[TextBox]:
    """OCR a frame region (upscaled by `scale`) and return text boxes in frame pixels."""
    x0, y0 = max(0, region[0]), max(0, region[1])
    gray = cv2.cvtColor(crop_region(frame, region), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    boxes = []
    for pts, text, conf in _get_reader().readtext(gray, allowlist=allowlist):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = (int(x0 + min(xs) / scale), int(y0 + min(ys) / scale),
               int((max(xs) - min(xs)) / scale), int((max(ys) - min(ys)) / scale))
        boxes.append((box, text.strip(), float(conf)))
    return boxes


def _read_text(frame: np.ndarray, region: tuple[int, int, int, int], allowlist: str) -> str:
    """OCR a small single-value region; EasyOCR needs tiny text upscaled."""
    scale = 64 / max(1, region[3])
    boxes = _read_boxes(frame, region, allowlist, scale)
    return boxes[0][1] if boxes else ""


def extract_ui_values(frame: np.ndarray, hud: HudState | None = None) -> dict[str, float | None]:
    """
    Run OCR on a single frame and return parsed UI values.

    Keys returned:
        gold, hp, level, stage, round, streak, xp_current, xp_needed

    streak is signed: positive for a win streak, negative for a loss streak.
    Values whose part of the HUD is not on screen (e.g. the bottom bar during
    carousel or while the shop is collapsed) are None rather than guessed.
    """
    return extract_ui_and_lobby(frame, hud)[0]


def extract_ui_and_lobby(
    frame: np.ndarray,
    hud: HudState | None = None,
) -> tuple[dict[str, float | None], list[float]]:
    """Same as extract_ui_values, plus the other players' HP (top to bottom)."""
    hud = hud or _get_hud().detect(frame)
    results: dict[str, float | None] = dict.fromkeys(
        ("gold", "hp", "level", "stage", "round", "streak", "xp_current", "xp_needed"))
    others: list[float] = []

    if hud.stage_visible:
        ix, iy = hud.stage_icon_xy
        dx, dy, w, h = _OCR_CFG["stage_text_offset"]
        parsed = parse_stage(_read_text(frame, (ix + dx, iy + dy, w, h), _ALLOWLIST["stage"]))
        if parsed:
            results["stage"], results["round"] = float(parsed[0]), float(parsed[1])
        results["hp"], others = read_hp(frame)

    if hud.shop_visible:
        for key, region in REGIONS.items():
            region = shift(region, hud.bottom_dx)
            text = _read_text(frame, region, _ALLOWLIST.get(key, _ALLOWLIST["default"]))
            if key == "xp":
                parsed = parse_xp(text)
                if parsed:
                    results["xp_current"], results["xp_needed"] = float(parsed[0]), float(parsed[1])
            else:
                results[key] = _parse_number(text)
        if results["streak"]:
            results["streak"] *= streak_sign(frame, hud.bottom_dx)

    return results, others


def streak_sign(frame: np.ndarray, dx: int = 0) -> int:
    """+1 if the streak flame is red (win streak), -1 if blue (loss streak)."""
    b, _, r = crop_region(frame, shift(_OCR_CFG["streak_icon"], dx)).reshape(-1, 3).mean(axis=0)
    return 1 if r >= b else -1


def read_hp(frame: np.ndarray) -> tuple[float | None, list[float]]:
    """
    Read HP from the right-hand player list.

    Returns (own_hp, other_players_hp). Others are listed top to bottom and
    include eliminated players as 0.

    When a panel has replaced the list (the "Damage Dealt" recap, or a unit's
    info panel) only the top band is read: the recap keeps the player's own
    HP there, and its damage numbers must not be taken for HP. Others are
    then unknown and returned empty.
    """
    if hp_panel_open(frame):
        boxes = _read_boxes(frame, tuple(_OCR_CFG["hp_top_band"]), _ALLOWLIST["default"], scale=2)
        return pick_hp(boxes)[0], []
    boxes = _read_boxes(frame, tuple(_OCR_CFG["hp_strip"]), _ALLOWLIST["default"], scale=2)
    return pick_hp(boxes)


def hp_panel_open(frame: np.ndarray) -> bool:
    """True if a panel covers the rows of the player list."""
    zone = crop_region(frame, tuple(_OCR_CFG["hp_panel_zone"]))
    return bool((zone.max(axis=2) < 45).mean() > _OCR_CFG["hp_panel_dark"])


def pick_hp(boxes: list[TextBox]) -> tuple[float | None, list[float]]:
    """
    Split player-list OCR boxes into the player's own HP and everyone else's.

    Own HP is the large number set further left; other players' HP is small
    text in a fixed column further right. Player names clipped by the strip
    edge produce junk boxes, which are rejected by position, size and value.
    """
    own_lo, own_hi = _OCR_CFG["hp_own_x"]
    min_h = _OCR_CFG["hp_own_min_height"]
    others_x = _OCR_CFG["hp_others_min_x"]

    def value(text: str) -> float | None:
        return float(text) if text.isdigit() and int(text) <= _OCR_CFG["hp_max"] else None

    others = [(y, value(t)) for (x, y, w, h), t, c in boxes
              if x >= others_x and c >= 0.5 and value(t) is not None]
    other_rows = [y for y, _ in others]

    own = [(h, value(t)) for (x, y, w, h), t, c in boxes
           if own_lo <= x <= own_hi and h >= min_h and c >= 0.5 and value(t) is not None
           # the own row has no small number, so a box sharing a row with one is junk
           and all(abs(y - oy) > 20 for oy in other_rows)]

    own_hp = max(own)[1] if own else None
    return own_hp, [v for _, v in sorted(others)]


def read_shop(frame: np.ndarray, hud: HudState | None = None) -> list[str | None]:
    """
    Read the champion name on each shop card, left to right.

    Returns one entry per slot in SHOP_SLOTS; None for an empty (bought) slot
    or a name that couldn't be matched to the champion list.
    """
    dx = (hud or _get_hud().detect(frame)).bottom_dx
    names = []
    for slot in SHOP_SLOTS:
        boxes = sorted(_read_boxes(frame, shift(slot, dx), _ALLOWLIST["shop"], scale=2))
        names.append(snap_name(" ".join(text for _, text, _ in boxes), CHAMPIONS))
    return names


def snap_name(text: str, known: list[str], cutoff: float = 0.6) -> str | None:
    """
    Snap OCR'd text to the closest known champion name.
    With no known list, returns the whitespace-normalised text as-is.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < 2:
        return None
    if not known:
        return cleaned
    by_key = {k.lower(): k for k in known}
    match = difflib.get_close_matches(cleaned.lower(), list(by_key), n=1, cutoff=cutoff)
    return by_key[match[0]] if match else None


def _parse_number(text: str) -> float | None:
    match = re.search(r"\d+", text)
    return float(match.group()) if match else None


def parse_stage(text: str) -> tuple[int, int] | None:
    """
    Parse stage text like '3-2' into (stage, round).
    Also handles '3.2' in case OCR reads the dash as a dot.
    """
    match = re.search(r"(\d+)[^0-9](\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_xp(text: str) -> tuple[int, int] | None:
    """Parse XP text like '0 / 4' or '2/6' into (current_xp, xp_to_level)."""
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
