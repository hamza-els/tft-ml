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
from src.vision.text_bank import EMPTY, text_bank

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


def field_region(key: str, hud: HudState) -> tuple[int, int, int, int]:
    """Where a bottom-bar number is on this frame: gold right of the gold icon found
    (gold_text_dx), the others at their calibrated place moved with the bar (bottom_dx)."""
    region = REGIONS[key]
    if key == "gold" and hud.gold_x is not None:
        x, y, w, h = region
        return hud.gold_x + int(_OCR_CFG["gold_text_dx"]), y, w, h
    return shift(region, hud.bottom_dx)


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


class OcrCache:
    """
    Skips OCR that would read the same thing as on the previous frame.

    Dedup only keeps a frame when something changed, but usually only one or
    two numbers did. Each small region (stage, gold, level, XP, streak, each
    shop card) is re-read only when its text pixels changed since the last
    time it was read (and until two reads agree); otherwise the last value is
    returned. Text pixels are the bright ones (grey > TEXT_LEVEL): scenery
    moving behind the HUD doesn't count as a change.

    HP is read once per round instead (`hp_needed`): it only changes when a
    fight ends, and the round counter moves on right after. The HP column
    itself is useless as a change signal: notices slide over the names and
    the portrait rings animate on almost every frame.

    One cache per game; call reset() between games.
    """
    TEXT_LEVEL = 170
    CHANGED_PIXELS = 12        # text pixels that must differ to count as a change

    def __init__(self):
        self.reset()

    HP_AGREE = 2               # own-HP readings that must agree before HP stops being read this round
    HP_MAX_READS = 4           # ...or give up after this many and take the most common
    HP_ROWS = 7                # ...and all seven other players were read (else the fullest list)

    def reset(self):
        self.seen: dict[str, tuple[np.ndarray, object, bool]] = {}   # key -> (text pixels, value, confirmed)
        self.hp_round: tuple | None = None
        self.hp: tuple[float | None, list[float]] = (None, [])
        self.hp_votes: list[float] = []
        self.hp_reads = 0
        self.reads = self.skips = 0

    def get(self, key: str, frame: np.ndarray, region: tuple[int, int, int, int], read):
        """
        The region's value: re-used while its text pixels stay the same, but
        only once two reads of the unchanged region agreed. OCR is not fully
        stable (the same "42" can come out as "12" once), so a single read is
        not trusted for a whole stretch of frames. (Reading the confirmation at
        a different upscale was tried: it did not catch the misreads, which
        both upscales share, and the smaller one added new ones like "142".)
        """
        gray = cv2.cvtColor(crop_region(frame, region), cv2.COLOR_BGR2GRAY)
        text = gray > self.TEXT_LEVEL
        old = self.seen.get(key)
        same = (old is not None and old[0].shape == text.shape
                and int((old[0] != text).sum()) < self.CHANGED_PIXELS)
        if same and old[2]:
            self.skips += 1
            return old[1]
        value = read()
        self.reads += 1
        self.seen[key] = (text, value, same and value == old[1])
        return value

    def read_hp(self, frame: np.ndarray, stage_round: tuple) -> tuple[float | None, list[float]]:
        """
        HP for this round, read on its first frames until two own-HP readings
        agree and all seven other players' HP were read (a panel can hide the list on the
        first frames), at most HP_MAX_READS times.
        """
        if None in stage_round:
            return self.hp
        if stage_round != self.hp_round:
            self.hp_round, self.hp, self.hp_votes, self.hp_reads = stage_round, (None, []), [], 0
        own, others = self.hp
        settled = own is not None and self.hp_votes.count(own) >= self.HP_AGREE and len(others) >= self.HP_ROWS
        if settled or self.hp_reads >= self.HP_MAX_READS:
            self.skips += 1
            return self.hp
        new_own, new_others = read_hp(frame)
        self.reads += 1
        self.hp_reads += 1
        if new_own is not None:
            self.hp_votes.append(new_own)
            own = max(set(self.hp_votes), key=self.hp_votes.count)
        # the fullest list of the round's reads (a row missed on one read is often there on the next)
        self.hp = (own, new_others if len(new_others) >= len(others) else others)
        return self.hp


def extract_ui_and_lobby(
    frame: np.ndarray,
    hud: HudState | None = None,
    cache: OcrCache | None = None,
) -> tuple[dict[str, float | None], list[float]]:
    """
    Same as extract_ui_values, plus the other players' HP (top to bottom).

    With a cache (consecutive frames of one game), unchanged regions are not
    read again and HP is read once per round (see OcrCache).
    """
    hud = hud or _get_hud().detect(frame)
    results: dict[str, float | None] = dict.fromkeys(
        ("gold", "hp", "level", "stage", "round", "streak", "xp_current", "xp_needed"))
    others: list[float] = []

    def read(key: str, region, allowlist: str) -> str:
        def look():
            # the text bank first (a picture seen before, src/vision/text_bank.py), else OCR
            bank = text_bank()
            if bank is not None:
                value = bank.lookup(key, cv2.cvtColor(crop_region(frame, region), cv2.COLOR_BGR2GRAY))
                if value:
                    return value
            return _read_text(frame, region, allowlist)
        if cache is None:
            return look()
        return cache.get(key, frame, region, look)

    if hud.stage_visible:
        ix, iy = hud.stage_icon_xy
        dx, dy, w, h = _OCR_CFG["stage_text_offset"]
        parsed = parse_stage(read("stage", (ix + dx, iy + dy, w, h), _ALLOWLIST["stage"]))
        if parsed:
            results["stage"], results["round"] = float(parsed[0]), float(parsed[1])
        if cache is None:
            results["hp"], others = read_hp(frame)
        else:
            results["hp"], others = cache.read_hp(frame, (results["stage"], results["round"]))

    if hud.shop_visible:
        for key in REGIONS:
            region = field_region(key, hud)
            text = read(key, region, _ALLOWLIST.get(key, _ALLOWLIST["default"]))
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
    own, others = pick_hp(boxes, rows=True)
    return (own[2] if own else None), [v for _, v in sorted(fill_hp_rows(frame, own, others))]


# Player list rows (2026-11-25): eight rows ROW_PITCH apart, the first number at HP_ROW0; the own
# row is taller and pushes the rows below it down by OWN_ROW_EXTRA. Measured on the 14 full-frame VODs.
HP_ROW0, HP_ROW_PITCH, OWN_ROW_EXTRA = 193, 80, 10
HP_ROW_BOX = (1820, -6, 44, 26)      # one row's number: x, y offset from the row, w, h


def fill_hp_rows(frame: np.ndarray, own, others: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """
    Rows the whole-strip OCR missed, read one by one (Hamza's checks, 2026-11-25: lobby HP right
    on 29 of 43 frames; 100s and faint rows went missing, and a row missing on the round's first
    reads stayed missing all round). Each missing row's number box is OCR'd alone at 4x; a value is
    kept only if it fits the list's order (players are listed by HP, highest first). Needs the own
    row (it fixes where the others are). On 151 random frames: all 7 others read on 70 -> 97;
    14 marked frames: 10 now equal to what the frame shows, 2 more where the frame differs from the mark.
    """
    if own is None:
        return others
    own_i = round((own[1] - HP_ROW0) / HP_ROW_PITCH)
    known = sorted(others + [(own[1], own[2])])
    out = list(others)
    for i in range(8):
        if i == own_i:
            continue
        y = HP_ROW0 + HP_ROW_PITCH * i + (OWN_ROW_EXTRA if i > own_i else 0)
        if any(abs(y - oy) <= 20 for oy, _ in out):
            continue
        x, dy, w, h = HP_ROW_BOX
        boxes = [(b, _hp_value(t), c) for b, t, c in _read_boxes(frame, (x, y + dy, w, h), _ALLOWLIST["default"], 4)]
        boxes = [b for b in boxes if b[1] is not None and b[2] >= 0.3]
        if not boxes:
            continue
        v = max(boxes, key=lambda b: b[2])[1]
        above = [hv for hy, hv in known if hy < y]
        below = [hv for hy, hv in known if hy > y]
        if (above and v > above[-1]) or (below and v < below[0]):
            continue                                   # out of order: a misread
        out.append((y, v))
        known = sorted(known + [(y, v)])
    return out


def _hp_value(text: str) -> float | None:
    """An HP number, or None. "1001": the portrait ring's edge read as a trailing 1 after 100."""
    if not text.isdigit():
        return None
    mx = _OCR_CFG["hp_max"]
    if int(text) <= mx:
        return float(text)
    if len(text) == 4 and int(text[:3]) <= mx:
        return float(text[:3])
    return None


def hp_panel_open(frame: np.ndarray) -> bool:
    """True if a panel covers the rows of the player list."""
    zone = crop_region(frame, tuple(_OCR_CFG["hp_panel_zone"])).astype(np.int16)
    diff = np.abs(zone - np.array(_OCR_CFG["hp_panel_color"], np.int16)).max(axis=2)
    return bool((diff < _OCR_CFG["hp_panel_tolerance"]).mean() > _OCR_CFG["hp_panel_share"])


def pick_hp(boxes: list[TextBox], rows: bool = False):
    """
    Split player-list OCR boxes into the player's own HP and everyone else's.

    Own HP is the large number set further left; other players' HP is small
    text in a fixed column further right. Player names clipped by the strip
    edge produce junk boxes, which are rejected by position, size and value.
    """
    own_lo, own_hi = _OCR_CFG["hp_own_x"]
    min_h = _OCR_CFG["hp_own_min_height"]
    others_x = _OCR_CFG["hp_others_min_x"]

    value = _hp_value

    # Rows are ~80 px apart. The win/loss mark under a player's HP sometimes
    # reads as "7"; keep only the most confident number in each row.
    others: list[tuple[int, float, float]] = []
    for (x, y, w, h), t, c in sorted(boxes, key=lambda b: -b[2]):
        if x >= others_x and c >= 0.5 and value(t) is not None \
                and all(abs(y - oy) > 30 for oy, _, _ in others):
            others.append((y, value(t), c))
    other_rows = [y for y, _, _ in others]

    own = [(h, y, value(t)) for (x, y, w, h), t, c in boxes
           if own_lo <= x <= own_hi and h >= min_h and c >= 0.5 and value(t) is not None
           # the own row has no small number, so a box sharing a row with one is junk
           and all(abs(y - oy) > 20 for oy in other_rows)]

    if rows:          # (height, y, value) of the own number, and (y, value) of the others
        return (max(own) if own else None), [(y, v) for y, v, _ in others]
    own_hp = max(own)[2] if own else None
    return own_hp, [v for _, v, _ in sorted(others)]


def read_shop(frame: np.ndarray, hud: HudState | None = None,
              cache: OcrCache | None = None) -> list[str | None]:
    """
    Read the champion name on each shop card, left to right.

    Returns one entry per slot in SHOP_SLOTS; None for an empty (bought) slot
    or a name that couldn't be matched to the champion list. With a cache,
    a card whose text didn't change since the last frame is not read again.
    """
    dx = (hud or _get_hud().detect(frame)).bottom_dx
    names = []
    for i, slot in enumerate(SHOP_SLOTS):
        region = shift(slot, dx)

        def read(region=region):
            bank = text_bank()
            if bank is not None:
                value = bank.lookup("shop", cv2.cvtColor(crop_region(frame, region), cv2.COLOR_BGR2GRAY))
                if value is not None:
                    return None if value == EMPTY else value
            boxes = sorted(_read_boxes(frame, region, _ALLOWLIST["shop"], scale=2))
            return snap_name(" ".join(text for _, text, _ in boxes), CHAMPIONS)
        names.append(read() if cache is None else cache.get(f"shop{i}", frame, region, read))
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
