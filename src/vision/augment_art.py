"""
Official augment art rendered to look like the in-game bar icon, as a template for
augments never seen in footage (Hamza's idea, 2026-11-23).

The art (Community Dragon, 64x64 on black) is shrunk to the icon's size, darkened
with a gamma curve, placed on the bar's dark panel colour and blurred. The few
settings were fitted on 572 labelled bar icons of 142 augments (2-fold split by
augment, scratchpad art_to_template.py): on augments left out of the fit the right
augment's render matched best on 93% of icons (97% in the top 8), against 62% / 83%
for the labeler's old feature comparison with the same art. The right render
scores lower than a footage template would (median 0.78, a footage medoid
~0.9+), hence its own threshold (`augments.art_match_threshold`).
"""

import cv2
import numpy as np

# fitted 2026-11-23 (both folds chose the same settings)
SIZE = 34            # px the 64 px art is shrunk to
DY = -1              # px up from centre
GAIN, GAMMA = 0.7, 1.4
BLUR = 1.2           # gaussian sigma
PANEL = (27, 30, 31)  # BGR of the bar's panel behind the icon (median of labelled crops' top rows)
SHAPE = (38, 37)     # icon crop (h, w) as AugmentReader cuts it


def render(art: np.ndarray, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    h, w = shape
    a = cv2.resize(art, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a = np.clip(a ** GAMMA * GAIN, 0, 1) * 255
    out = np.empty((h, w, 3), np.float32)
    out[:] = PANEL
    y0, x0 = (h - SIZE) // 2 + DY, (w - SIZE) // 2
    ys, xs = slice(max(y0, 0), min(y0 + SIZE, h)), slice(max(x0, 0), min(x0 + SIZE, w))
    patch = a[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    opaque = patch.max(axis=2, keepdims=True) > 12          # the art's black background shows the panel
    out[ys, xs] = np.where(opaque, patch, out[ys, xs])
    out = cv2.GaussianBlur(out, (0, 0), BLUR)
    return np.clip(out, 0, 255).astype(np.uint8)
