"""
Read HUD text by picture instead of OCR, where the picture was seen before.

The HUD numbers (stage, gold, level, streak, XP) and the shop card names are
drawn on solid panels, so one value draws almost the same bright pixels every
time. A bank keeps a few typical text masks (grey > LEVEL) per value, cut from
footage whose OCR reading agreed with the cleaned value
(scripts/build_text_bank.py). A crop is read from the bank when its nearest
mask is within the field's tolerance (pixels that differ) and no mask of
another value is; otherwise None and the caller runs OCR as before.

A crop with almost no text (< INK pixels) is left to OCR, except a shop card
with no text even at grey DIM: an empty (bought) slot.

Held out by VOD (2026-11-23, 14 full-frame VODs, 150 frames a game, scored
against OCR): answered 75-96% of crops per field, 99.85-100% the same as OCR.
"""

from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
BANK_DIR = _ROOT / "assets" / "templates" / "text_bank"
LEVEL = 170           # grey level of text pixels (as ocr_pipeline.OcrCache)
INK = 20              # fewer text pixels than this: not read from the bank
DIM = 80              # a shop card is empty only if it has no text at this grey level either: cards fading in
                      # show their names dimmer than LEVEL (empty slots: 0 pixels above 80 on 95%; fading names ~150)
EMPTY = ""            # a shop card with no text: an empty slot
_POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


class TextBank:
    def __init__(self, folder: Path = BANK_DIR):
        self.fields: dict[str, dict] = {}
        for path in sorted(folder.glob("*.npz")):
            z = np.load(path, allow_pickle=False)
            packed = z["packed"]
            self.fields[path.stem] = {"packed": packed, "values": [str(v) for v in z["values"]],
                                      "ink": _POP[packed].sum(axis=1).astype(np.int32),
                                      "shape": tuple(int(x) for x in z["shape"]), "tol": int(z["tol"])}
        self.hits = self.misses = 0

    def lookup(self, field: str, gray: np.ndarray) -> str | None:
        """The value the crop shows, EMPTY for an empty shop card, or None (run OCR)."""
        bank = self.fields.get(field)
        if bank is None or gray.shape != bank["shape"]:
            return None
        mask = gray > LEVEL
        ink = int(mask.sum())
        if ink < INK:
            if field == "shop" and int((gray > DIM).sum()) < INK:
                self.hits += 1
                return EMPTY
            return None
        # masks whose pixel count differs by more than the tolerance can't be within it
        tol = bank["tol"]
        cand = np.flatnonzero(np.abs(bank["ink"] - ink) <= tol)
        if len(cand) == 0:
            self.misses += 1
            return None
        d = _POP[np.bitwise_xor(bank["packed"][cand], np.packbits(mask.ravel())[None, :])].sum(axis=1)
        close = cand[d <= tol]
        if len(close) == 0:
            self.misses += 1
            return None
        values = {bank["values"][k] for k in close}
        if len(values) > 1:                  # two values look alike here: OCR decides
            self.misses += 1
            return None
        self.hits += 1
        return values.pop()


_BANK: TextBank | None = None


def text_bank() -> TextBank | None:
    """The bank, if built (None: OCR only)."""
    global _BANK
    if _BANK is None:
        _BANK = TextBank() if BANK_DIR.exists() and any(BANK_DIR.glob("*.npz")) else False
    return _BANK or None
