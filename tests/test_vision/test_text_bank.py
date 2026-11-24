"""
HUD text read by picture (src/vision/text_bank.py).

Usage:
    python -m pytest tests/test_vision/test_text_bank.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.vision.text_bank import EMPTY, TextBank


def _bank(tmp_path, masks, values, tol=8, field="gold"):
    m = np.stack([x.ravel() for x in masks]).astype(bool)
    np.savez(tmp_path / f"{field}.npz", packed=np.packbits(m, axis=1), values=np.array(values),
             shape=np.array(masks[0].shape), tol=np.array(tol))
    return TextBank(tmp_path)


def _glyph(cols):
    g = np.zeros((10, 20), bool)
    for c in cols:
        g[2:8, c] = True
    return g


def test_reads_a_known_picture_and_tolerates_noise(tmp_path):
    a, b = _glyph([2, 3, 4, 5]), _glyph([10, 11, 12, 13])
    bank = _bank(tmp_path, [a, b], ["12", "47"])
    noisy = a.copy()
    noisy[0, 0] = noisy[9, 19] = True
    assert bank.lookup("gold", np.where(noisy, 255, 0).astype(np.uint8)) == "12"
    assert bank.lookup("gold", np.where(b, 255, 0).astype(np.uint8)) == "47"


def test_unknown_or_ambiguous_picture_goes_to_ocr(tmp_path):
    a, b = _glyph([2, 3, 4, 5]), _glyph([2, 3, 4, 6])           # 12 pixels apart: both within 8 of a middle
    bank = _bank(tmp_path, [a, b], ["12", "17"], tol=8)
    far = _glyph([15, 16, 17, 18])
    assert bank.lookup("gold", np.where(far, 255, 0).astype(np.uint8)) is None
    mid = a.copy()
    mid[2:5, 6] = True
    mid[2:5, 5] = False
    assert bank.lookup("gold", np.where(mid, 255, 0).astype(np.uint8)) is None
    assert bank.lookup("gold", np.zeros((10, 20), np.uint8)) is None     # no text: OCR


def test_blank_shop_card_is_empty_but_dim_text_is_not(tmp_path):
    bank = _bank(tmp_path, [_glyph([2, 3, 4, 5])], ["Ahri"], field="shop")
    assert bank.lookup("shop", np.zeros((10, 20), np.uint8)) == EMPTY
    dim = np.where(_glyph([2, 3, 4, 5]), 120, 0).astype(np.uint8)       # a card fading in
    assert bank.lookup("shop", dim) is None
