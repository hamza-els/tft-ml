"""
Label a whole champion at once - which unit, its star level and its items -
with every model's answer pre-filled, and keep score of how often each model
was right.

Usage:
    python scripts/label_champions.py              # unit crops not reviewed here yet, in random order
    python scripts/label_champions.py --unsure     # least confident unit guesses first

Each review feeds all three models' training data:
    champion -> data/processed/unit_crops/labels.json   (unit classifier)
    star     -> data/processed/unit_crops/star_labels.json
    items    -> data/processed/item_crops/labels.json   (item classifier; icon crops written like
                collect_item_crops.py; empty slots are labelled "not an item")
and the models' pre-filled answers next to the final ones in
data/processed/unit_crops/champion_reviews.json, which the scoreboard (and
scripts/train_all.py) read. Random order (the default) keeps the scoreboard an
honest estimate of each model's accuracy.

Keys:
    Enter          accept everything as shown (or the first search result, if typing)
    1-8 / F1-F4    pick a search result / a champion guess
    Ctrl+1/2/3     star level
    Alt+1/2/3      change item slot 1/2/3 (search, "no item", "can't tell")
    Ctrl+D         no items
    Tab            not a unit        Shift+Tab   unit, can't tell which
    Ctrl+T         covered by a tooltip
    Esc            skip   Ctrl+Z  undo   Ctrl+Q  quit (everything is saved immediately)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import re
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

from scripts.label_common import load, merge_save, search_champions as fuzzy, to_photo
from src.state.game_data import load_game_data
from src.utils import paths
from src.vision.bench import BenchSlots
from src.vision.board import BoardHexes
from src.vision.unit_bar import UnitItemReader, bench_bar, board_bar, crop, held_count, star_level
from src.vision.unit_classifier import UnitClassifier

NONE, UNKNOWN, COVERED = "_none", "_unknown", "_covered"
SPECIAL = {NONE: "not a unit", UNKNOWN: "unit, can't tell which", COVERED: "covered by a tooltip"}
STAR_LABELS = paths.CROP_DIR / "star_labels.json"
REVIEWS = paths.CROP_DIR / "champion_reviews.json"
ITEM_PAD = 4
BG, FG, DIM, ACCENT, GOOD, BAD = "#1e1f24", "#e8e8ea", "#8b8d98", "#f2c14e", "#5fbf7f", "#d96b6b"


def scoreboard(reviews: dict) -> dict:
    """How often each model's pre-filled answer was kept, over the given reviews."""
    s = {"champion": [0, 0], "star": [0, 0], "item count": [0, 0], "item names": [0, 0]}
    for r in reviews.values():
        pred, final = r["pred"], r["final"]
        s["champion"][0] += pred["name"] == final["name"]
        s["champion"][1] += 1
        if final["name"].startswith("_"):
            continue
        if final.get("star"):
            s["star"][0] += pred.get("star") == final["star"]
            s["star"][1] += 1
        pi, fi = pred.get("items", []), final.get("items", [])
        s["item count"][0] += len(pi) == len(fi)
        s["item count"][1] += 1
        for a, b in zip(pi, fi):
            if b not in (None, UNKNOWN):
                s["item names"][0] += a == b
                s["item names"][1] += 1
    return s


class ChampionLabeler:
    def __init__(self, unsure: bool):
        self.game = load_game_data()
        self.unit_names = self.game.unit_names
        self.item_names = sorted(set(self.game.items) | set(self.game.emblems))
        self.classifier = UnitClassifier()
        self.items = UnitItemReader()
        self.board, self.bench = BoardHexes(), BenchSlots()
        self.index = load(paths.CROP_DIR / "index.json")
        reviews = load(REVIEWS)
        files = [n for n in self.index if (paths.CROP_IMAGES / n).exists() and n not in reviews]
        random.seed()
        random.shuffle(files)
        if unsure:
            cache = load(paths.CROP_SCORE_CACHE).get("scores", {})
            files.sort(key=lambda f: cache.get(f, 1.0))
        self.queue = files
        self.pos = 0
        self.history: list[str] = []
        self.session: dict = {}
        self.changes = {"labels": {}, "stars": {}, "reviews": {}, "item_labels": {}, "item_index": {}}
        self._build_window()
        self.show()

    # ── models ────────────────────────────────────────────────────────────────

    def predict(self, name: str):
        """(frame, bar, crop image, champion guesses, star, item names) for a unit crop."""
        info = self.index[name]
        frame = cv2.imread(info["frame_path"])
        img = cv2.imread(str(paths.CROP_IMAGES / name))
        if frame is None or img is None:
            return None
        m = re.search(r"__([hs]\d+)\.jpg$", name)
        pos = m.group(1) if m else None     # propagated / rider / pair crops: no bar of their own position
        bar = (None if pos is None else board_bar(frame, self.board, int(pos[1]), int(pos[2])) if pos[0] == "h"
               else bench_bar(frame, self.bench, int(pos[1:])))
        guesses = self.classifier.predict(img, top_k=4)
        star = star_level(frame, bar) if bar else None
        held = []
        if bar:
            slots = self.items.slots_many(frame, [bar])[0]
            held = [s["name"] for s in slots[:held_count(slots)]]
        return frame, bar, img, guesses, star, held

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        r = self.root = tk.Tk()
        r.title("Label champions")
        r.configure(bg=BG)
        self.big, self.small = tkfont.Font(size=13), tkfont.Font(size=10)
        self.status = tk.Label(r, bg=BG, fg=DIM, font=self.small, anchor="w", justify="left")
        self.status.grid(row=0, column=0, columnspan=3, sticky="we", padx=12, pady=(10, 4))
        self.crop_view = tk.Label(r, bg=BG)
        self.crop_view.grid(row=1, column=0, rowspan=6, padx=12, sticky="n")
        self.zoom_view = tk.Label(r, bg=BG)
        self.zoom_view.grid(row=1, column=1, padx=6, sticky="nw")
        self.context_view = tk.Label(r, bg=BG)
        self.context_view.grid(row=1, column=2, padx=6, sticky="nw")

        tk.Label(r, text="Champion:", bg=BG, fg=DIM, font=self.small).grid(row=2, column=1, sticky="w", padx=6)
        self.guess_row = tk.Frame(r, bg=BG)
        self.guess_row.grid(row=3, column=1, columnspan=2, sticky="w", padx=6)
        panel = tk.Frame(r, bg=BG)
        panel.grid(row=4, column=1, columnspan=2, sticky="we", padx=6, pady=4)
        self.query = tk.StringVar()
        self.entry = tk.Entry(panel, textvariable=self.query, font=self.big, width=28, bg="#2b2d35", fg=FG,
                              insertbackground=FG, relief="flat")
        self.entry.pack(side="left", ipady=4)
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(r, bg=BG)
        self.result_box.grid(row=5, column=1, columnspan=2, sticky="we", padx=6)

        line = tk.Frame(r, bg=BG)
        line.grid(row=6, column=1, columnspan=2, sticky="we", padx=6, pady=6)
        tk.Label(line, text="Star:", bg=BG, fg=DIM, font=self.small).pack(side="left")
        self.star_buttons = []
        for s in (1, 2, 3):
            b = tk.Button(line, text=f"{s}", width=3, font=self.big, relief="flat", command=lambda s=s: self.set_star(s))
            b.pack(side="left", padx=2)
            self.star_buttons.append(b)
        tk.Label(line, text="   Items:", bg=BG, fg=DIM, font=self.small).pack(side="left")
        self.item_buttons = []
        for k in range(3):
            b = tk.Button(line, text="-", font=self.small, relief="flat", width=22, anchor="w",
                          command=lambda k=k: self.edit_item(k))
            b.pack(side="left", padx=2)
            self.item_buttons.append(b)

        actions = tk.Frame(r, bg=BG)
        actions.grid(row=7, column=0, columnspan=3, sticky="we", padx=12, pady=(4, 12))
        for text, cmd in (("Accept (Enter)", self.accept), ("Not a unit (Tab)", lambda: self.special(NONE)),
                          ("Can't tell (Shift+Tab)", lambda: self.special(UNKNOWN)),
                          ("Tooltip (Ctrl+T)", lambda: self.special(COVERED)), ("No items (Ctrl+D)", self.clear_items),
                          ("Skip (Esc)", self.skip), ("Undo (Ctrl+Z)", self.undo), ("Quit (Ctrl+Q)", self.quit)):
            tk.Button(actions, text=text, command=cmd, font=self.small, bg="#33353f", fg=FG, relief="flat",
                      padx=8, pady=3).pack(side="left", padx=(0, 6))
        r.bind("<Return>", lambda e: self.enter())
        r.bind("<Tab>", lambda e: (self.special(NONE), "break")[1])
        r.bind("<Shift-Tab>", lambda e: (self.special(UNKNOWN), "break")[1])
        try:
            r.bind("<ISO_Left_Tab>", lambda e: (self.special(UNKNOWN), "break")[1])
        except tk.TclError:
            pass
        r.bind("<Control-t>", lambda e: (self.special(COVERED), "break")[1])
        r.bind("<Control-d>", lambda e: (self.clear_items(), "break")[1])
        r.bind("<Escape>", lambda e: self.skip())
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-q>", lambda e: self.quit())
        for n in range(1, 9):
            r.bind(str(n), lambda e, k=n: (self.pick_result(k - 1), "break")[1])
        for n in range(1, 5):
            r.bind(f"<F{n}>", lambda e, k=n: self.pick_guess(k - 1))
        for s in (1, 2, 3):
            r.bind(f"<Control-Key-{s}>", lambda e, s=s: (self.set_star(s), "break")[1])
            r.bind(f"<Alt-Key-{s}>", lambda e, k=s - 1: (self.edit_item(k), "break")[1])
        r.protocol("WM_DELETE_WINDOW", self.quit)

    def _buttons(self, parent, names, prefix, scores=None, chosen=None):
        for c in parent.winfo_children():
            c.destroy()
        for i, n in enumerate(names):
            text = SPECIAL.get(n, n) + (f"  {scores[n]:.0%}" if scores and n in scores else "")
            tk.Button(parent, text=f"{prefix}{i + 1}  {text}", anchor="w", font=self.big, relief="flat",
                      bg="#2b2d35", fg=ACCENT if n == chosen else FG, padx=8, pady=2,
                      command=lambda n=n: self.set_champion(n)) \
                .pack(side="left" if parent is self.guess_row else "top",
                      fill=None if parent is self.guess_row else "x", padx=(0, 6), pady=1)

    def refresh(self):
        self._buttons(self.guess_row, [g for g, _ in self.cur["guesses"]], "F", dict(self.cur["guesses"]),
                      chosen=self.cur["name"])
        for s, b in zip((1, 2, 3), self.star_buttons):
            b.configure(bg=ACCENT if self.cur["star"] == s else "#33353f", fg="#000" if self.cur["star"] == s else FG)
        for k, b in enumerate(self.item_buttons):
            v = self.cur["items"][k] if k < len(self.cur["items"]) else None
            b.configure(text=f"{k + 1}: {v if v else ('can' + chr(39) + 't tell' if v == '' else '-')}",
                        bg="#33353f", fg=FG)

    def refresh_results(self):
        self.results = fuzzy(self.query.get(), self.unit_names)
        self._buttons(self.result_box, self.results, "")

    # ── flow ──────────────────────────────────────────────────────────────────

    def show(self):
        while self.pos < len(self.queue):
            name = self.queue[self.pos]
            pred = self.predict(name)
            if pred is not None:
                break
            self.queue.pop(self.pos)
        else:
            self.status.configure(text="All done. Close the window.")
            return
        frame, bar, img, guesses, star, held = pred
        self.cur = {"crop": name, "frame": frame, "bar": bar, "guesses": guesses,
                    "name": guesses[0][0] if guesses else UNKNOWN, "star": star, "items": list(held)}
        self.pred = {"name": self.cur["name"], "star": star, "items": list(held)}
        self.crop_photo = to_photo(cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
        self.crop_view.configure(image=self.crop_photo)
        if bar is not None:
            z = crop(frame, (bar.left - 32, bar.top - 16, 130, 56))
            self.zoom_photo = to_photo(cv2.resize(z, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
            self.zoom_view.configure(image=self.zoom_photo)
        ctx = frame.copy()
        x, y, w, h = self.index[name]["window"]
        cv2.rectangle(ctx, (x, y), (x + w, y + h), (0, 255, 255), 4)
        self.context_photo = to_photo(cv2.resize(ctx, (480, 270), interpolation=cv2.INTER_AREA))
        self.context_view.configure(image=self.context_photo)
        self.query.set("")
        self.refresh_results()
        self.refresh()
        self.update_status()
        self.entry.focus_set()

    def update_status(self):
        def fmt(s):
            return "   ".join(f"{k} {a}/{n} ({a / n:.0%})" if n else f"{k} -" for k, (a, n) in s.items())
        allr = dict(load(REVIEWS))
        allr.update({k: v for k, v in self.changes["reviews"].items() if v})
        self.status.configure(
            text=f"{len(self.queue) - self.pos} left   |   reviewed this session {len(self.session)}\n"
                 f"models right, this session:  {fmt(scoreboard(self.session))}\n"
                 f"models right, all reviews:   {fmt(scoreboard(allr))}")

    def set_champion(self, n):
        self.cur["name"] = n
        self.query.set("")
        self.refresh()

    def pick_guess(self, i):
        if i < len(self.cur["guesses"]):
            self.set_champion(self.cur["guesses"][i][0])

    def pick_result(self, i):
        if i < len(self.results):
            self.set_champion(self.results[i])

    def set_star(self, s):
        self.cur["star"] = s
        self.refresh()

    def clear_items(self):
        self.cur["items"] = []
        self.refresh()

    def edit_item(self, k):
        """Small search window for one item slot; items fill from the left."""
        top = tk.Toplevel(self.root, bg=BG)
        top.title(f"Item slot {k + 1}")
        q = tk.StringVar()
        e = tk.Entry(top, textvariable=q, font=self.big, width=28, bg="#2b2d35", fg=FG, insertbackground=FG)
        e.pack(padx=8, pady=8)
        box = tk.Frame(top, bg=BG)
        box.pack(fill="x", padx=8)

        def choose(value):
            items = list(self.cur["items"])
            if value is None:                       # no item here (and none after it)
                items = items[:k]
            else:
                while len(items) < k:
                    items.append("")                # an earlier slot holds something unread
                if k < len(items):
                    items[k] = value
                else:
                    items.append(value)
            self.cur["items"] = items
            top.destroy()
            self.refresh()
            self.entry.focus_set()

        def refresh(*_):
            for c in box.winfo_children():
                c.destroy()
            for n in fuzzy(q.get(), self.item_names)[:8]:
                tk.Button(box, text=n, anchor="w", font=self.small, bg="#2b2d35", fg=FG, relief="flat",
                          command=lambda n=n: choose(n)).pack(fill="x", pady=1)
        q.trace_add("write", refresh)
        tk.Button(top, text="No item (and none after)", command=lambda: choose(None), bg="#33353f", fg=FG,
                  relief="flat").pack(side="left", padx=8, pady=8)
        tk.Button(top, text="Item, can't tell", command=lambda: choose(""), bg="#33353f", fg=FG,
                  relief="flat").pack(side="left", padx=8, pady=8)
        e.bind("<Return>", lambda ev: choose(fuzzy(q.get(), self.item_names)[0]) if fuzzy(q.get(), self.item_names) else None)
        e.bind("<Escape>", lambda ev: top.destroy())
        e.focus_set()

    def enter(self):
        if self.query.get().strip():
            if self.results:
                self.set_champion(self.results[0])
            return
        self.accept()

    def special(self, label):
        self.cur["name"] = label
        self.accept()

    def accept(self):
        if self.pos >= len(self.queue):
            return
        c = self.cur
        name = c["crop"]
        final = {"name": c["name"]}
        self.changes["labels"][name] = c["name"]
        if not c["name"].startswith("_"):
            final["star"] = c["star"]
            final["items"] = [i if i else UNKNOWN for i in c["items"]]
            if c["star"]:
                self.changes["stars"][name] = c["star"]
            if c["bar"] is not None:
                self._item_labels(name, final["items"])
        review = {"pred": {**self.pred, "items": [i if i else UNKNOWN for i in self.pred["items"]]}, "final": final}
        self.changes["reviews"][name] = review
        self.session[name] = review
        self.history.append(name)
        self.save()
        self.pos += 1
        self.show()

    def _item_labels(self, name, items):
        """Item crops for all three slots: the item, or "not an item" for empty slots."""
        frame, bar = self.cur["frame"], self.cur["bar"]
        for slot, region in enumerate(bar.item_regions()):
            x, y, w, h = region
            icon = crop(frame, (x - ITEM_PAD, y - ITEM_PAD, w + 2 * ITEM_PAD, h + 2 * ITEM_PAD))
            if icon.shape[:2] != (h + 2 * ITEM_PAD, w + 2 * ITEM_PAD):
                continue
            iname = name[:-4] + f"__i{slot}.png"
            paths.ITEM_IMAGES.mkdir(parents=True, exist_ok=True)
            paths.ITEM_CONTEXT.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(paths.ITEM_IMAGES / iname), icon)
            cv2.imwrite(str(paths.ITEM_CONTEXT / iname), crop(frame, (bar.left - 32, bar.top - 16, 130, 56)))
            self.changes["item_index"][iname] = {"frame_path": self.index[name]["frame_path"],
                                                 "region": [int(v) for v in region], "unit_crop": name,
                                                 "slot": slot, "guess": None, "score": 0.0, "outline": 0.0}
            self.changes["item_labels"][iname] = items[slot] if slot < len(items) else NONE

    def save(self):
        merge_save(paths.CROP_DIR / "labels.json", self.changes["labels"])
        merge_save(STAR_LABELS, self.changes["stars"])
        merge_save(REVIEWS, self.changes["reviews"])
        merge_save(paths.ITEM_DIR / "index.json", self.changes["item_index"])
        merge_save(paths.ITEM_DIR / "labels.json", self.changes["item_labels"])

    def skip(self):
        if self.pos < len(self.queue):
            self.queue.append(self.queue.pop(self.pos))
            self.show()

    def undo(self):
        if not self.history:
            return
        name = self.history.pop()
        for key in ("labels", "stars", "reviews"):
            self.changes[key][name] = None
        for slot in range(3):
            iname = name[:-4] + f"__i{slot}.png"
            if iname in self.changes["item_labels"]:
                self.changes["item_labels"][iname] = None
        self.session.pop(name, None)
        self.save()
        self.pos = self.queue.index(name) if name in self.queue else self.pos
        self.show()

    def quit(self):
        self.save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsure", action="store_true", help="Least confident unit guesses first")
    args = parser.parse_args()
    app = ChampionLabeler(args.unsure)
    app.root.mainloop()
    print(f"Reviewed {len(app.session)} champions this session.")


if __name__ == "__main__":
    main()
