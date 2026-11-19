"""
Small helpers shared by the labeling apps (label_units, label_items,
label_champions, label_augments): fuzzy name search, image -> Tk photo, and
label files saved by merging.

Kept free of torch: label_augments.py only needs these, and importing them
from label_units.py pulled in the unit classifier, torch and torchvision
(~60 s of a ~70 s start-up on 2026-11-19).
"""

import base64
import difflib
import json
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

MAX_RESULTS = 8


def search_champions(query: str, names: list[str], limit: int = MAX_RESULTS) -> list[str]:
    """
    Names ranked by how well they match a possibly misspelled query.

    A name that starts with the query ranks first, then one that contains it,
    then names by spelling similarity (so "morgna" finds Morgana and "khazik"
    finds Kha'Zix). Punctuation, spaces and case are ignored.
    """
    def norm(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    q = norm(query)
    if not q:
        return []
    scored = []
    for name in names:
        n = norm(name)
        ratio = difflib.SequenceMatcher(None, q, n).ratio()
        # Compare against the start of the name too: people type the first few letters.
        prefix_ratio = difflib.SequenceMatcher(None, q, n[:len(q) + 1]).ratio()
        score = max(ratio, prefix_ratio * 0.95)
        if n.startswith(q):
            score += 2.0
        elif q in n:
            score += 1.0
        scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for score, name in scored[:limit] if score >= 0.45]


def to_photo(img: np.ndarray) -> tk.PhotoImage:
    ok, buf = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def merge_save(path: Path, changes: dict):
    """Write only this session's changes into the file as it is now (None = remove)."""
    data = load(path)
    for k, v in changes.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=0, sort_keys=True), encoding="utf-8")
