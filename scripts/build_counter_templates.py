"""
Build templates for the board counter ("units / team size", the big
translucent "8/8" in the middle of the arena during planning) from footage.

Usage:
    python scripts/build_counter_templates.py

Uses cleaned states on the local disk (~/tft_data/state_vectors/*/*.clean.jsonl)
and their frames. For each value n, the frames where the cleaned board fills
exactly n slots at level n most likely show "n/n"; the edge image of the
counter is cut out on both sides of the slash, aligned and median-combined,
so the arena behind it averages out:

    assets/templates/board_counter/left_<n>.png    units on board (left of the slash)
    assets/templates/board_counter/right_<n>.png   team size (right of the slash)

10 comes from level-10 frames (right side) and level-10 frames with a full
board (left side), plus frames checked by eye to show "10/10". Only crops
that agree with the median are kept. 0, 1, 11 and 12 are not built (early
game or rare).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import json
import random

import cv2
import numpy as np

from src.state.game_data import load_game_data
from src.utils import paths
from src.vision.board_counter import CFG, OUT_DIR, edges, side

PER_VALUE = 150


AGREE = 0.45      # crops that correlate less with the median are left out (covered, or another number)


def aligned_median(crops: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """
    Median after shifting each crop onto the running median (text jitters a
    few px), keeping only the crops that agree with it: frames where a unit
    or the Little Legend covers the counter, or that show another number
    than assumed, would blur the template.
    """
    med = np.median(crops, axis=0)
    kept = len(crops)
    for _ in range(4):
        m = 8
        out = []
        for c in crops:
            res = cv2.matchTemplate(c, med[m:-m, m:-m], cv2.TM_CCOEFF_NORMED)
            _, s, _, (dx, dy) = cv2.minMaxLoc(res)
            if s >= AGREE:
                out.append(cv2.warpAffine(c, np.float32([[1, 0, m - dx], [0, 1, m - dy]]), c.shape[::-1]))
        if len(out) >= 10:
            med, kept = np.median(out, axis=0), len(out)
    return med, kept


def main():
    game = load_game_data()
    full = collections.defaultdict(list)     # n -> frames with level n and a full board of n
    level = collections.defaultdict(list)    # n -> frames with level n
    for p in sorted(paths.STATES.glob("*/game_*.clean.jsonl")):
        vod, gname = p.parent.name, p.name[:7]
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not (r.get("planning") and r.get("home_board") and r.get("level")):
                continue
            n = int(r["level"])
            frame = Path.home() / "tft_frames_temp" / vod / gname / f"frame_{r['frame']:06d}.jpg"
            level[n].append(frame)
            if sum(game.slots(u["name"]) for u in r["board"]) == n:
                full[n].append(frame)
            # Checked by eye: this game shows "10/10" at level 9 from stage 5 on
            # (a team-size augment). Level-10 frames alone are too few for "10".
            if (vod, gname) == ("ev6dfSvJuJ8", "game_03") and n == 9 and (r.get("stage") or 0) >= 5:
                level[10].append(frame)
                full[10].append(frame)

    random.seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in range(2, 11):
        for which, source in (("left", full[n]), ("right", full[n] if n < 10 else level[n])):
            frames = random.sample(source, min(PER_VALUE, len(source)))
            crops = []
            for f in frames:
                img = cv2.imread(str(f))
                if img is not None:
                    crops.append(side(edges(img), which, two_digits=n >= 10))
            if len(crops) < 10:
                print(f"  {which}_{n}: only {len(crops)} frames, skipped")
                continue
            med, kept = aligned_median(crops)
            cv2.imwrite(str(OUT_DIR / f"{which}_{n}.png"), np.clip(med * 255, 0, 255).astype(np.uint8))
            print(f"  {which}_{n}: {kept} of {len(crops)} frames agree")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
