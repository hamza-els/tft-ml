import numpy as np

from src.vision import augment_overlay as ov


class _NoOcr:
    def readtext(self, *a, **k):
        raise AssertionError("OCR called on a box that is not shown")


class _Fake:
    def __init__(self):
        self.calls = 0

    def readtext(self, *a, **k):
        self.calls += 1
        return []


def _frame(aug=False, portal=False):
    img = np.full((1080, 1920, 3), (60, 64, 58), np.uint8)       # dark HUD
    for on, (x, y, w, h) in ((aug, ov.AUG_BOX), (portal, ov.PORTAL_BOX)):
        if on:
            img[y:y + h, x:x + w] = ov.BEIGE.astype(np.uint8)
            img[y + 10:y + 20, x + 10:x + w - 10] = 30                # a line of dark text
    return img


def test_no_overlay_skips_ocr():
    img = _frame()
    assert ov.shown(img, ov.AUG_BOX) < ov.SHOWN_MIN
    assert ov.read(img, _NoOcr()) == (None, [])


def test_each_box_ocrd_only_when_shown():
    r = _Fake()
    ov.read(_frame(portal=True), r)          # sponsor pop-up over the augments box, portal still up
    assert r.calls == 1
    r = _Fake()
    ov.read(_frame(aug=True, portal=True), r)
    assert r.calls == 2
    assert ov.shown(_frame(aug=True), ov.AUG_BOX) > 0.5
