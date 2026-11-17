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
right it is someone else's. `home_board_rows` uses the bar first and falls
back to the arena colour only when no bar is on screen (early game, or the
bar hidden).

Planning phase, two signs that must agree (`planning_rows`):
  - The round timer bar under the stage bar drains during planning and
    refills when combat starts. Within one round, frames before the first
    refill are planning; everything after is combat.
  - A player can only place units on their own half, so during planning the
    far part of the board has no health bars (`is_planning`). This covers
    rounds whose planning frames were not captured, where the timer alone
    would mistake the whole round for planning.
"""

import numpy as np

from src.utils.config import vision_cfg
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
        if row.get("away_augments"):
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
    out = []
    current_round, last_fill, in_combat = None, None, False
    for row in rows:
        key = (row.get("stage"), row.get("round_num"))
        fill = row.get("timer_fill")
        if key != current_round:
            current_round, last_fill, in_combat = key, None, False
        if fill is not None:
            if last_fill is not None and fill - last_fill >= refill:
                in_combat = True
            last_fill = fill
        out.append(bool(row.get("planning")) and not in_combat and None not in key)
    return out


def is_planning(frame: np.ndarray, cfg: dict | None = None) -> bool:
    """True if no health bars are drawn on the far (enemy) half of the board."""
    cfg = cfg or vision_cfg["arena"]
    x, y, w, h = cfg["enemy_half"]
    green, red = health_bar_pixels(frame[y:y + h, x:x + w])
    return green + red < int(cfg["max_bar_pixels"])
