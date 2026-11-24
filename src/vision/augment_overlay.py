"""
Read a game's augments from BoxBox's stream overlay instead of the icons.

His overlay (bottom left, above the shop) shows "Portal: <name>" and, under it,
"Augments: <name>, <name>, ...": the real names with tiers ("Jeweled Lotus I",
"Salvage Bin+"), in a beige box with dark text. A sponsor pop-up covers it now
and then, so a game is read from many frames and the readings are voted on
(scripts/read_augment_overlay.py).

OCR of the box is rough: commas get lost, lines split mid-name, "+" comes out as
"t". So the text is not split on commas. Its words are matched against the
set's augment names (assets/game_data/<set>_augments.json): the sequence of
known names that best covers the words in reading order wins (dynamic
programming over words, difflib similarity per name, "+" also tried as "t";
words that fit no name are skipped at a cost). A qualifier in brackets
("Magic Roll (Emblem)") is dropped first: the set data has none.

Region: 1920x1080 frames of the calibrated layout; the "Augments:" box sits
at about x 15-285, y 955-1075, the "Portal:" line at y 885-930.
"""

import difflib
import re
from functools import lru_cache

import cv2
import numpy as np

from src.state.game_data import load_game_data

AUG_BOX = (12, 950, 280, 128)      # x, y, w, h
PORTAL_BOX = (12, 880, 280, 52)
MIN_SIM = 0.72                    # a word span must be this like a name to count as it
SKIP_COST = 0.6                   # per word left unmatched (characters' worth, scaled below)
# The box is beige (BGR ~181, 208, 239). On 2,775 picked frames of the 14 full-frame VODs
# (2026-11-24): every frame OCR read augments on had >= 23% of AUG_BOX beige, every
# portal read >= 51% of PORTAL_BOX; frames without the box mostly < 5%. At 10%
# no OCR reading was lost and 628 augment-box / 115 portal OCRs were skipped.
BEIGE = np.array([181, 208, 239], np.int16)
BEIGE_TOL = 30
SHOWN_MIN = 0.10
ALLOW ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.'+:()!-&"


def _words(text: str) -> list[str]:
    text = re.sub(r"\([^)]*\)?", " ", text)          # "(Emblem)", "(Kobuko)", a cut-off "(Emb"
    return re.findall(r"[a-z0-9'+!&]+", text.lower())


@lru_cache(maxsize=1)
def _names() -> list[tuple[str, str, str, int]]:
    """(name, words joined, the same with '+' read as 't', word count) for every augment of the set."""
    out = []
    for name in load_game_data().augments:
        w = _words(name)
        joined = " ".join(w)
        out.append((name, joined, joined.replace("+", "t"), len(w)))
    return out


def parse(text: str) -> list[str]:
    """Augment names in the order the text gives them (text after 'Augments:' if present)."""
    m = re.search(r"ugment[s5]?\s*[:;.]?", text, re.I)
    words = _words(text[m.end():] if m else text)
    words = [w for w in words if w]
    n = len(words)
    if not n:
        return []
    names = _names()
    best = [(-1e9, None, None)] * (n + 1)              # score, previous index, name
    best[0] = (0.0, None, None)
    for i in range(n):
        if best[i][0] <= -1e9:
            continue
        score_i = best[i][0]
        # skip word i
        cand = score_i - SKIP_COST * len(words[i])
        if cand > best[i + 1][0]:
            best[i + 1] = (cand, i, None)
        for name, joined, joined_t, k in names:
            for j in range(i + max(1, k - 1), min(n, i + k + 1) + 1):
                span = " ".join(words[i:j])
                sim = max(difflib.SequenceMatcher(None, span, joined).ratio(),
                          difflib.SequenceMatcher(None, span, joined_t).ratio())
                if sim >= MIN_SIM:
                    cand = score_i + sim * len(joined) - (1 - sim) * len(joined)
                    if cand > best[j][0]:
                        best[j] = (cand, i, name)
    out, j = [], n
    while j > 0 and best[j][1] is not None:
        if best[j][2]:
            out.append(best[j][2])
        j = best[j][1]
    return out[::-1]


def _ocr_text(img: np.ndarray, box: tuple[int, int, int, int], reader) -> str:
    x, y, w, h = box
    crop = cv2.resize(img[y:y + h, x:x + w], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(crop, allowlist=ALLOW)
    # reading order: by line (box centre y, 24 px bands at 2x), then x
    items = []
    for b, text, _ in results:
        ys = [p[1] for p in b]
        xs = [p[0] for p in b]
        items.append((round((min(ys) + max(ys)) / 2 / 24), min(xs), text))
    return " ".join(t for _, _, t in sorted(items))


def shown(img: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Share of the box in the overlay's beige (every channel within BEIGE_TOL of BEIGE)."""
    x, y, w, h = box
    crop = img[y:y + h, x:x + w].astype(np.int16)
    return float((np.abs(crop - BEIGE).max(axis=2) < BEIGE_TOL).mean())


def read(img: np.ndarray, reader) -> tuple[str | None, list[str]]:
    """
    (portal text or None, augment names in overlay order) from one frame. A box is
    OCR'd only when it is shown (>= SHOWN_MIN beige): newer streams have the overlay
    off (htwxJLXZE_g, OGLPl4JkAmQ after its first game), and a sponsor pop-up often
    covers the augments box while the portal line still shows.
    """
    augs, portal = [], None
    if shown(img, AUG_BOX) >= SHOWN_MIN:
        text = _ocr_text(img, AUG_BOX, reader)
        augs = parse(text) if re.search(r"ugment", text, re.I) else []
    if shown(img, PORTAL_BOX) >= SHOWN_MIN:
        m = re.search(r"ortal\s*[:;.]?\s*(.+)", _ocr_text(img, PORTAL_BOX, reader), re.I)
        portal = m.group(1).strip() if m else None
    return portal, augs
