"""
Interactive region calibrator for vision_config.yaml.

Usage:
    python scripts/calibrate_regions.py data/raw/frames/<VOD_ID>/frame_000100.jpg

Controls:
    Mouse move  — crosshair + live (x, y) readout
    Left-click  — print exact pixel coordinate
    Right-click — mark a corner; right-click a second corner to print the
                  (x, y, w, h) region tuple, ready to paste into the config
    R           — reset corner selection
    Q / Esc     — quit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/calibrate_regions.py <path_to_frame.jpg>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    print("Left-click  → print coordinate")
    print("Right-click → first corner, then second corner → prints (x, y, w, h)")
    print("R → reset corners | Q/Esc → quit")

    state = {"corners": [], "cursor": (0, 0)}

    def draw(x, y):
        canvas = img.copy()
        # Crosshair
        cv2.line(canvas, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.line(canvas, (0, y), (w, y), (0, 255, 0), 1)
        # Coordinate label — flip to left side near the right edge
        label = f"({x}, {y})"
        lx = x + 8 if x < w - 120 else x - 120
        ly = y - 8 if y > 20 else y + 20
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        # First corner marker
        if state["corners"]:
            cx, cy = state["corners"][0]
            cv2.drawMarker(canvas, (cx, cy), (0, 128, 255),
                           cv2.MARKER_CROSS, 16, 2)
            # Preview rectangle
            rx, ry = min(cx, x), min(cy, y)
            rw, rh = abs(x - cx), abs(y - cy)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 128, 255), 1)
            cv2.putText(canvas, f"w={rw} h={rh}", (rx + 4, ry - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, cv2.LINE_AA)
        cv2.imshow("Calibrator", canvas)

    def on_mouse(event, x, y, flags, _):
        state["cursor"] = (x, y)
        draw(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"  point : ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if not state["corners"]:
                state["corners"] = [(x, y)]
                print(f"  corner 1 : ({x}, {y})  — right-click second corner")
            else:
                x1, y1 = state["corners"][0]
                rx, ry = min(x1, x), min(y1, y)
                rw, rh = abs(x - x1), abs(y - y1)
                print(f"  region   : ({rx}, {ry}, {rw}, {rh})  # x, y, w, h")
                state["corners"] = []

    cv2.namedWindow("Calibrator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrator", min(w, 1280), min(h, 720))
    cv2.setMouseCallback("Calibrator", on_mouse)
    draw(0, 0)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            state["corners"] = []
            x, y = state["cursor"]
            draw(x, y)
            print("  corners reset")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
