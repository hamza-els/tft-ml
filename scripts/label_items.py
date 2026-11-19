"""
Label item icons (cut by scripts/collect_item_crops.py) with the item they show.

Usage:
    python scripts/label_items.py                  # unlabelled, grouped by the matcher's guess
    python scripts/label_items.py --unsure         # least certain matches first
    python scripts/label_items.py --review         # labelled ones again
    python scripts/label_items.py --relabel "Fae Emblem"

How it works:
  - The icon is shown large on the left; on the right, the unit's health bar
    and item row it came from, for context.
  - "Looks like" buttons are the template matcher's best guesses with their
    scores (assets/templates/unit_items/). With the box empty, Enter accepts
    the first. Unnamed template IDs (uitem_XXX) can be picked too.
  - Type any part of the item's name for a fuzzy search over the set's items.
  - By default crops come in runs of 5 with the same guess, cycling through
    every item, so each run is mostly Enter, Enter, Enter and every item is
    covered early.

Keys:
    Enter        first search result (or first guess when the box is empty)
    1-8          that search result
    F1-F4        that guess
    Tab          not an item (empty slot, scenery)
    Shift+Tab    an item, but it can't be told which
    Ctrl+T       covered by a tooltip
    Esc          skip for now
    Ctrl+Z       undo
    Ctrl+Q       quit (every label is saved immediately)

Labels: data/processed/item_crops/labels.json, {crop file name: item name};
"_none", "_unknown", "_covered" as above.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

from scripts.label_units import search_champions as fuzzy_search, to_photo
from src.state.game_data import load_game_data
from src.utils import paths
from src.vision.unit_bar import UnitItemReader

NONE, UNKNOWN, COVERED = "_none", "_unknown", "_covered"
SPECIAL = {NONE: "not an item", UNKNOWN: "item, can't tell which", COVERED: "covered by a tooltip"}
MAX_RESULTS, MAX_GUESSES = 8, 4
RUN = 5                  # crops with the same guess shown in a row before moving to the next item
PAD = 4
BG, FG, DIM, ACCENT = "#1e1f24", "#e8e8ea", "#8b8d98", "#f2c14e"


class ItemLabeler:
    def __init__(self, mode: str, relabel: str | None):
        game = load_game_data()
        self.reader = UnitItemReader()
        template_names = {n for n in self.reader.names}
        self.names = sorted(set(game.items) | set(game.emblems) | template_names)
        self.index = json.loads((paths.ITEM_DIR / "index.json").read_text(encoding="utf-8"))
        self.labels_path = paths.ITEM_DIR / "labels.json"
        self.labels: dict[str, str] = json.loads(self.labels_path.read_text(encoding="utf-8")) \
            if self.labels_path.exists() else {}
        self.changed: dict[str, str | None] = {}

        files = [n for n in self.index if (paths.ITEM_IMAGES / n).exists()]
        if relabel is not None:
            queue = [f for f in files if self.labels.get(f) == relabel]
        elif mode == "review":
            queue = [f for f in files if f in self.labels]
        else:
            queue = [f for f in files if f not in self.labels]
        if mode == "unsure":
            queue.sort(key=lambda f: self.index[f].get("score", 0))
        elif relabel is None and mode != "review":
            # Runs of RUN crops with the same guess, cycling through every
            # guess: quick to confirm, and every item is covered early on.
            groups: dict[str, list[str]] = {}
            for f in sorted(queue, key=lambda f: -self.index[f].get("score", 0)):
                groups.setdefault(str(self.index[f].get("guess")), []).append(f)
            queue, k = [], 0
            while any(groups.values()):
                for g in sorted(groups):
                    queue.extend(groups[g][k:k + RUN])
                k += RUN
                groups = {g: v for g, v in groups.items() if len(v) > k}
        self.queue, self.total = queue, len(files)
        self.history: list[str] = []
        self.pos = 0
        self.guesses: list[str] = []
        self.guess_scores: dict[str, float] = {}
        self._build_window()
        self.show()

    # ── data ──────────────────────────────────────────────────────────────────

    def guess(self, icon: np.ndarray) -> list[tuple[str, float]]:
        """Matcher's best templates for a padded icon (best over small shifts), one per name."""
        size = icon.shape[0] - 2 * PAD
        best: dict[str, float] = {}
        for dx in range(-2, 3):
            for dy in (-1, 0, 1):
                c = icon[PAD + dy:PAD + dy + size, PAD + dx:PAD + dx + size]
                if c.shape[:2] != (size, size):
                    continue
                for name, s in zip(self.reader.names, self.reader.scores(c)):
                    if s > best.get(name, -1):
                        best[name] = float(s)
        return sorted(best.items(), key=lambda kv: -kv[1])[:MAX_GUESSES]

    def save(self):
        try:
            on_disk = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            on_disk = {}
        for name, label in self.changed.items():
            if label is None:
                on_disk.pop(name, None)
            else:
                on_disk[name] = label
        self.labels_path.write_text(json.dumps(on_disk, indent=0, sort_keys=True), encoding="utf-8")

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Label items")
        self.root.configure(bg=BG)
        self.big, self.small = tkfont.Font(size=13), tkfont.Font(size=10)
        self.status = tk.Label(self.root, bg=BG, fg=DIM, font=self.small, anchor="w")
        self.status.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))
        self.icon_view = tk.Label(self.root, bg=BG)
        self.icon_view.grid(row=1, column=0, rowspan=4, padx=12, pady=6, sticky="n")
        self.context_view = tk.Label(self.root, bg=BG)
        self.context_view.grid(row=1, column=1, padx=12, pady=6, sticky="nw")
        tk.Label(self.root, text="Looks like:", bg=BG, fg=DIM, font=self.small, anchor="w") \
            .grid(row=2, column=1, sticky="we", padx=12)
        self.guess_row = tk.Frame(self.root, bg=BG)
        self.guess_row.grid(row=3, column=1, sticky="we", padx=12)
        panel = tk.Frame(self.root, bg=BG)
        panel.grid(row=4, column=1, sticky="nwe", padx=12, pady=8)
        self.query = tk.StringVar()
        self.entry = tk.Entry(panel, textvariable=self.query, font=self.big, width=30,
                              bg="#2b2d35", fg=FG, insertbackground=FG, relief="flat")
        self.entry.pack(fill="x", ipady=6)
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(panel, bg=BG)
        self.result_box.pack(fill="x", pady=(6, 0))
        actions = tk.Frame(self.root, bg=BG)
        actions.grid(row=5, column=0, columnspan=2, sticky="we", padx=12, pady=(4, 12))
        for text, command in (("Not an item  (Tab)", lambda: self.assign(NONE)),
                              ("Item, can't tell  (Shift+Tab)", lambda: self.assign(UNKNOWN)),
                              ("Covered by tooltip  (Ctrl+T)", lambda: self.assign(COVERED)),
                              ("Skip  (Esc)", self.skip), ("Undo  (Ctrl+Z)", self.undo), ("Quit  (Ctrl+Q)", self.quit)):
            tk.Button(actions, text=text, command=command, font=self.small, bg="#33353f", fg=FG,
                      activebackground="#444754", activeforeground=FG, relief="flat", padx=10, pady=4) \
                .pack(side="left", padx=(0, 8))
        r = self.root
        r.bind("<Return>", lambda e: self.pick_first())
        r.bind("<Tab>", lambda e: (self.assign(NONE), "break")[1])
        r.bind("<Shift-Tab>", lambda e: (self.assign(UNKNOWN), "break")[1])
        try:
            r.bind("<ISO_Left_Tab>", lambda e: (self.assign(UNKNOWN), "break")[1])
        except tk.TclError:
            pass
        r.bind("<Control-t>", lambda e: (self.assign(COVERED), "break")[1])
        r.bind("<Escape>", lambda e: self.skip())
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-q>", lambda e: self.quit())
        for n in range(1, MAX_RESULTS + 1):
            r.bind(str(n), lambda e, k=n: (self.pick(self.results, k - 1), "break")[1])
        for n in range(1, MAX_GUESSES + 1):
            r.bind(f"<F{n}>", lambda e, k=n: self.pick(self.guesses, k - 1))
        r.protocol("WM_DELETE_WINDOW", self.quit)

    def _buttons(self, parent, names, prefix, scores=None):
        for child in parent.winfo_children():
            child.destroy()
        for i, name in enumerate(names):
            text = SPECIAL.get(name, name)
            if name.startswith("uitem_"):
                text += "  (unnamed)"
            if scores and name in scores:
                text += f"  {scores[name]:.2f}"
            tk.Button(parent, text=f"{prefix}{i + 1}  {text}", anchor="w", command=lambda n=name: self.assign(n),
                      font=self.big, bg="#2b2d35", fg=ACCENT if i == 0 else FG, activebackground="#444754",
                      activeforeground=FG, relief="flat", padx=10, pady=3) \
                .pack(fill="x" if parent is self.result_box else None,
                      side="top" if parent is self.result_box else "left", padx=(0, 6), pady=2)

    # ── flow ──────────────────────────────────────────────────────────────────

    def show(self):
        while self.pos < len(self.queue):
            name = self.queue[self.pos]
            icon = cv2.imread(str(paths.ITEM_IMAGES / name))
            if icon is not None:
                break
            self.queue.pop(self.pos)
        else:
            self.status.config(text=f"All done - {len(self.labels)} of {self.total} labelled. Close the window.")
            self.icon_view.config(image="")
            self.context_view.config(image="")
            return
        self.icon_photo = to_photo(cv2.resize(icon, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST))
        self.icon_view.config(image=self.icon_photo)
        ctx = cv2.imread(str(paths.ITEM_CONTEXT / name))
        if ctx is not None:
            slot = self.index[name].get("slot", 0)
            # box the icon inside the context crop (bar.left - 32, bar.top - 16)
            x0, y0 = 32 - 9 + 26 * slot, 16 + 12
            cv2.rectangle(ctx, (x0 - 1, y0 - 1), (x0 + 26, y0 + 26), (0, 255, 255), 1)
            self.context_photo = to_photo(cv2.resize(ctx, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            self.context_view.config(image=self.context_photo)
        g = self.guess(icon)
        self.guesses = [n for n, _ in g]
        self.guess_scores = dict(g)
        self._buttons(self.guess_row, self.guesses, "F", self.guess_scores)
        current = self.labels.get(name)
        self.status.config(text=f"{len(self.labels)} of {self.total} labelled   |   {len(self.queue) - self.pos} left"
                                + (f"   |   current label: {current}" if current else ""))
        self.query.set("")
        self.refresh_results()
        self.entry.focus_set()

    def refresh_results(self):
        self.results = fuzzy_search(self.query.get(), self.names)
        self._buttons(self.result_box, self.results, "")

    def assign(self, label: str):
        if self.pos >= len(self.queue):
            return
        name = self.queue[self.pos]
        self.labels[name] = label
        self.changed[name] = label
        self.history.append(name)
        self.save()
        self.pos += 1
        self.show()

    def pick_first(self):
        if self.results:
            self.assign(self.results[0])
        elif not self.query.get().strip() and self.guesses:
            self.assign(self.guesses[0])

    def pick(self, options, i):
        if i < len(options):
            self.assign(options[i])

    def skip(self):
        if self.pos < len(self.queue):
            self.queue.append(self.queue.pop(self.pos))
            self.show()

    def undo(self):
        if not self.history:
            return
        name = self.history.pop()
        self.labels.pop(name, None)
        self.changed[name] = None
        self.save()
        if name in self.queue:
            self.pos = self.queue.index(name)
        else:
            self.queue.insert(self.pos, name)
        self.show()

    def quit(self):
        self.save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true", help="Show already-labelled crops")
    parser.add_argument("--unsure", action="store_true", help="Least certain matches first")
    parser.add_argument("--relabel", metavar="LABEL", help="Every crop with this label again")
    args = parser.parse_args()
    if not (paths.ITEM_DIR / "index.json").exists():
        sys.exit("No item crops yet - run scripts/collect_item_crops.py first")
    mode = "review" if args.review else "unsure" if args.unsure else "new"
    app = ItemLabeler(mode, args.relabel)
    app.root.mainloop()
    print(f"{len(app.labels)} of {app.total} item crops labelled -> {app.labels_path}")


if __name__ == "__main__":
    main()
