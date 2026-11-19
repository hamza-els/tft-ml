"""
Tells whether a frame shows the player's own board, and whether it is the
planning phase.

Own board: every player has a different arena skin, and the camera leaves the
player's own arena whenever they fight on an opponent's board or scout. The
HUD (gold, shop, HP) stays the player's own the whole time, but the units on
screen are then somebody else's. Each frame gets a small colour signature from
patches of scenery around the board; within one game the most common signature
is the player's own arena, and frames far from it are away or scouting.
(Round-transition frames, where the arena is dimmed, also come out as "not
home" — they are not useful for reading units either.)

The augment bar is a stronger sign when it is showing (see
src/vision/augments.py): on the left it is the player's own board, on the
right it is someone else's. In combat both players' bars show at once, the
board owner's on the left and the visitor's on the right. `home_board_rows`
uses the bar first and falls back to the arena colour when no bar is on
screen (early game, or the bar hidden) or both are (combat).

Planning phase, signs that must all agree (`planning_rows`):
  - The round timer bar under the stage bar drains during planning and
    refills when combat starts. Within one round, frames before the first
    refill are planning; everything after is combat. A frame with the timer
    empty is the instant combat starts and does not count.
  - A player can only place units on their own half, so during planning the
    far part of the board has no health bars (`is_planning`). This covers
    rounds whose planning frames were not captured, where the timer alone
    would mistake the whole round for planning.
  - The "Combat" banner is not up (`combat_banner`, also in `is_planning`):
    it shows for the first second or two of a fight, when units already face
    the enemy but enemy health bars may not be drawn yet.
"""

from pathlib import Path

import cv2
import numpy as np

from src.utils.config import vision_cfg

_ROOT = Path(__file__).resolve().parents[2]
from src.vision.board import health_bar_pixels


def arena_signature(frame: np.ndarray, cfg: dict | None = None) -> list[float]:
    """Mean colour of each scenery patch, flattened (3 numbers per patch)."""
    cfg = cfg or vision_cfg["arena"]
    sig: list[float] = []
    for x, y, w, h in cfg["patches"]:
        sig.extend(float(v) for v in frame[y:y + h, x:x + w].reshape(-1, 3).mean(axis=0))
    return [round(v, 1) for v in sig]


def home_board_mask(signatures: list, cfg: dict | None = None) -> list[bool]:
    """True where a frame's signature is close to the game's most common arena."""
    cfg = cfg or vision_cfg["arena"]
    known = [s for s in signatures if s]
    if not known:
        return [False] * len(signatures)
    median = np.median(np.array(known, dtype=np.float32), axis=0)
    limit = float(cfg["max_distance"])
    return [bool(s) and float(np.abs(np.array(s) - median).mean()) <= limit for s in signatures]


def home_board_rows(rows: list[dict], cfg: dict | None = None) -> list[bool]:
    """
    For one game's state rows: True where the player's own board is on screen.

    Rows need `augments` / `away_augments` (augment bar) and `arena_sig`.
    Rows from before those fields existed count as home.
    """
    sigs = [r.get("arena_sig") for r in rows]
    by_colour = home_board_mask(sigs, cfg) if any(sigs) else [True] * len(rows)
    out = []
    for row, colour in zip(rows, by_colour):
        if row.get("away_augments") and row.get("augments"):
            # In combat both players' bars show (the board owner's on the left, the
            # visitor's on the right), so the bars don't say whose board it is.
            out.append(colour)
        elif row.get("away_augments"):
            out.append(False)
        elif row.get("augments"):
            out.append(True)
        else:
            out.append(colour)
    return out


def timer_fill(frame: np.ndarray, cfg: dict | None = None) -> float:
    """How full the round timer bar is, 0-1 (turquoise pixels along the bar)."""
    cfg = cfg or vision_cfg["arena"]
    x, y, w, h = cfg["timer_bar"]
    px = frame[y:y + h, x:x + w].astype(np.int16)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    lit = ((g > 170) & (b > 160) & (r < 150)).mean(axis=0) > 0.5   # per column
    return round(min(1.0, float(lit.sum()) / float(cfg["timer_bar_length"])), 3)


def planning_rows(rows: list[dict], cfg: dict | None = None) -> list[bool]:
    """
    For one game's state rows in frame order: True where the frame is in the
    planning phase (timer not yet refilled this round, and no health bars on
    the far part of the board).
    """
    cfg = cfg or vision_cfg["arena"]
    refill = float(cfg["timer_refill"])
    empty = float(cfg["timer_empty"])
    out = []
    current_round, last_fill, in_combat = None, None, False
    for row in rows:
        key = (row.get("stage"), row.get("round_num"))
        fill = row.get("timer_fill")
        if key != current_round:
            current_round, last_fill, in_combat = key, None, False
        # An empty timer is the instant planning ends: units are already
        # turning to face the enemy.
        ending = fill is not None and fill <= empty
        # A reading at (near) zero is also what a hidden timer looks like: the
        # augment choice covers it at the start of 3-2 and 4-2. Such a reading
        # must not be the "before" of a refill, or the timer reappearing
        # (0.0 -> 0.85) reads as combat starting and the whole round's planning
        # is lost (86 of 254 x-2 rounds had no planning frame). A real combat
        # start still shows: the timer drains gradually, so the jump from its
        # last visible value (e.g. 0.10 -> 0.52) is a refill.
        if fill is not None and not ending:
            if last_fill is not None and fill - last_fill >= refill:
                in_combat = True
            last_fill = fill
        out.append(bool(row.get("planning")) and not in_combat and not ending and None not in key)
    return out


_COMBAT_WORD: np.ndarray | None = None


def combat_banner(frame: np.ndarray, cfg: dict | None = None) -> bool:
    """
    True while the "Combat" banner (crossed swords over the word "Combat") is
    shown under the stage bar: the first second or two of a fight, when the
    units have turned to face the enemy but no enemy bars are up yet.

    Matched on the word only (greyscale, median of 117 banners): the swords
    animate in, and a "Planning" banner with a different icon appears at the
    same spot when planning starts. Measured on 1,638 frames with a banner or
    bright text there: "Combat" banners score 0.6-1.0 (0.8+ unless fading),
    "Planning" banners 0.5-0.6, everything else lower.
    """
    global _COMBAT_WORD
    cfg = cfg or vision_cfg["arena"]
    if _COMBAT_WORD is None:
        _COMBAT_WORD = cv2.imread(str(_ROOT / cfg["combat_word"]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x, y, w, h = cfg["combat_banner_zone"]
    zone = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    score = float(cv2.matchTemplate(zone, _COMBAT_WORD, cv2.TM_CCOEFF_NORMED).max())
    return score >= float(cfg["combat_banner_score"])


def is_planning(frame: np.ndarray, cfg: dict | None = None) -> bool:
    """
    True if no health bars are drawn on the far (enemy) half of the board and
    the "Combat" banner is not up.
    """
    cfg = cfg or vision_cfg["arena"]
    x, y, w, h = cfg["enemy_half"]
    green, red = health_bar_pixels(frame[y:y + h, x:x + w])
    return green + red < int(cfg["max_bar_pixels"]) and not combat_banner(frame, cfg)
