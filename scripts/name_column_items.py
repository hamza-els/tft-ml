"""
Name the unnamed item-column templates (assets/templates/items_seen/item_XXX.png).

The item column (unequipped items, left edge) is read only by template
matching. 168 of its templates had no name on 2026-11-21, and they made up
41% of all column readings (item_001, Sparring Gloves with a stack badge,
alone 29%), all "unknown" to the dataset. Naming the most-read few fixes most
of that, so templates come most-read first.

Usage:
    python scripts/name_column_items.py            # label (one window)
    python scripts/name_column_items.py --apply    # rename the templates, rewrite bench_items in the
                                                   # local state files (then re-clean / encode / upload)

Window: the template large, the item column of a frame where it was read,
and up to 8 suggestions (official art and already-named templates, best
match first) with their icons.

Keys:
    Enter        first search result (or the first suggestion when the box is empty)
    1-8          that search result
    F1-F8        that suggestion
    Tab          not an item (scenery, a cursor, a tooltip edge)
    Shift+Tab    an item, but it can't be told which
    Esc          skip for now
    Ctrl+Z       undo
    Ctrl+Q       quit (every answer is saved immediately)

Answers: assets/templates/items_seen/names_review.json, {template: item name | "_none" | "_unknown"}.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import shutil
import tkinter as tk
from collections import Counter
from tkinter import font as tkfont

import cv2
import numpy as np

from scripts.label_common import search_champions as fuzzy_search, to_photo
from src.state.game_data import load_game_data
from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.items import ItemColumnReader

SEEN = paths.REPO / vision_cfg["item_column"]["seen_dir"]
OFFICIAL = paths.REPO / vision_cfg["template_matching"]["item_dir"]
ANSWERS = SEEN / "names_review.json"
REMOVED = SEEN.parent / "items_seen_removed"
NONE, UNKNOWN = "_none", "_unknown"
MAX_GUESSES, MAX_RESULTS = 8, 8
BG, FG, DIM, ACCENT = "#1e1f24", "#e6e6e6", "#9a9aa5", "#7fd1ff"


def base_name(name: str) -> str:
    return name.split("~")[0]


def load_answers() -> dict[str, str]:
    try:
        return json.loads(ANSWERS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def readings() -> tuple[Counter, dict[str, tuple[str, str, int]]]:
    """How often each column template was read in the local state files, and one (vod, game, frame) each."""
    counts: Counter = Counter()
    where: dict[str, tuple[str, str, int]] = {}
    for f in sorted(paths.STATES.glob("*/game_??.jsonl")):
        vod, game = f.parent.name, f.stem
        has_frames = (paths.FRAMES / vod / game).exists()
        for line in f.read_text(encoding="utf-8").splitlines():
            if '"item_' not in line:
                continue
            r = json.loads(line)
            for n in r.get("bench_items") or []:
                if n and n.startswith("item_"):
                    counts[n] += 1
                    if has_frames and n not in where:
                        where[n] = (vod, game, r["frame"])
    return counts, where


class Namer:
    def __init__(self):
        game = load_game_data()
        self.reader = ItemColumnReader()
        self.names = sorted(set(game.items) | set(game.emblems))
        self.answers = load_answers()
        self.counts, self.where = readings()
        unnamed = [p.stem for p in SEEN.glob("item_*.png")]
        self.total_reads = sum(self.counts[n] for n in unnamed)
        self.queue = sorted((n for n in unnamed if n not in self.answers), key=lambda n: -self.counts[n])
        self.n_templates = len(unnamed)
        self.history: list[str] = []
        self.pos = 0
        self.icons: dict[str, np.ndarray] = {}
        self._build_window()
        self.show()

    # ── data ──────────────────────────────────────────────────────────────────

    def suggestions(self, icon: np.ndarray) -> list[tuple[str, float]]:
        """Best item names for a template: official art and named footage templates, best score per name."""
        best: dict[str, float] = {}
        for tset, named in ((self.reader.official, False), (self.reader.seen, True)):
            if not tset.names:
                continue
            scores = np.stack(tset._rows) @ tset.prepare(icon)
            for name, s in zip(tset.names, scores):
                if named and name.startswith("item_"):
                    continue
                key = base_name(name)
                if key in self.names and s > best.get(key, -1.0):
                    best[key] = float(s)
        return sorted(best.items(), key=lambda kv: -kv[1])[:MAX_GUESSES]

    def icon_of(self, name: str) -> np.ndarray | None:
        if name not in self.icons:
            img = cv2.imread(str(OFFICIAL / f"{name}.png"))
            if img is None:
                img = cv2.imread(str(SEEN / f"{name}.png"))
            self.icons[name] = None if img is None else cv2.resize(img, (36, 36), interpolation=cv2.INTER_AREA)
        return self.icons[name]

    def context(self, name: str) -> np.ndarray | None:
        """The item column of a frame where this template was read, with the matching slot boxed."""
        if name not in self.where:
            return None
        vod, game, n = self.where[name]
        frame = cv2.imread(str(paths.FRAMES / vod / game / f"frame_{n:06d}.jpg"))
        if frame is None:
            return None
        r = self.reader
        tmpl = cv2.imread(str(SEEN / f"{name}.png"))
        for i in range(r.slots):
            x, y, w, h = r.icon_region(i)
            icon = frame[y:y + h, x:x + w]
            if tmpl is not None and icon.shape[:2] == (h, w) and \
                    float(r.seen.prepare(icon) @ r.seen.prepare(tmpl)) >= r.seen_threshold:
                cv2.rectangle(frame, (x - 3, y - 3), (x + w + 2, y + h + 2), (0, 255, 255), 2)
        y0 = max(0, r.y - 40)
        y1 = min(frame.shape[0], int(r.y + r.slots * r.pitch + 40))
        crop = frame[y0:y1, 0:r.x + r.size + 260]
        return cv2.resize(crop, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)

    def save(self, name: str, label: str | None):
        on_disk = load_answers()
        if label is None:
            on_disk.pop(name, None)
        else:
            on_disk[name] = label
        ANSWERS.write_text(json.dumps(on_disk, indent=1, sort_keys=True), encoding="utf-8")
        self.answers = on_disk

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Name item-column templates")
        self.root.configure(bg=BG)
        self.big, self.small = tkfont.Font(size=13), tkfont.Font(size=10)
        self.status = tk.Label(self.root, bg=BG, fg=DIM, font=self.small, anchor="w", justify="left")
        self.status.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))
        self.icon_view = tk.Label(self.root, bg=BG)
        self.icon_view.grid(row=1, column=0, padx=12, pady=6, sticky="n")
        self.context_view = tk.Label(self.root, bg=BG, fg=DIM, font=self.small)
        self.context_view.grid(row=1, column=1, rowspan=3, padx=12, pady=6, sticky="nw")
        tk.Label(self.root, text="Looks like (F1-F8):", bg=BG, fg=DIM, font=self.small, anchor="w") \
            .grid(row=2, column=0, sticky="we", padx=12)
        self.guess_box = tk.Frame(self.root, bg=BG)
        self.guess_box.grid(row=3, column=0, sticky="nwe", padx=12)
        panel = tk.Frame(self.root, bg=BG)
        panel.grid(row=4, column=0, columnspan=2, sticky="nwe", padx=12, pady=8)
        tk.Label(panel, text="Search (Enter = first result):", bg=BG, fg=DIM, font=self.small, anchor="w").pack(fill="x")
        self.query = tk.StringVar()
        self.entry = tk.Entry(panel, textvariable=self.query, font=self.big, width=34,
                              bg="#2b2d35", fg=FG, insertbackground=FG, relief="flat")
        self.entry.pack(fill="x", ipady=6)
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(panel, bg=BG)
        self.result_box.pack(fill="x", pady=(6, 0))
        actions = tk.Frame(self.root, bg=BG)
        actions.grid(row=5, column=0, columnspan=2, sticky="we", padx=12, pady=(4, 12))
        for text, command in (("Not an item  (Tab)", lambda: self.assign(NONE)),
                              ("Item, can't tell  (Shift+Tab)", lambda: self.assign(UNKNOWN)),
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
        self._photos = getattr(self, "_photos", {})
        self._photos[id(parent)] = []
        for i, name in enumerate(names):
            text = f"{prefix}{i + 1}  {name}" + (f"  {scores[name]:.2f}" if scores and name in scores else "")
            icon = self.icon_of(name)
            photo = to_photo(icon) if icon is not None else None
            if photo is not None:
                self._photos[id(parent)].append(photo)
            tk.Button(parent, text=text, image=photo, compound="left", anchor="w",
                      command=lambda n=name: self.assign(n), font=self.big, bg="#2b2d35",
                      fg=ACCENT if i == 0 else FG, activebackground="#444754", activeforeground=FG,
                      relief="flat", padx=8, pady=2).pack(fill="x", pady=1)

    # ── flow ──────────────────────────────────────────────────────────────────

    def show(self):
        if self.pos >= len(self.queue):
            self.status.config(text=f"All done - {len(self.answers)} templates answered. Close the window, "
                                    f"then tell Claude (or run --apply).")
            self.icon_view.config(image="")
            self.context_view.config(image="", text="")
            return
        name = self.queue[self.pos]
        icon = cv2.imread(str(SEEN / f"{name}.png"))
        self.icon_photo = to_photo(cv2.resize(icon, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
        self.icon_view.config(image=self.icon_photo)
        ctx = self.context(name)
        if ctx is not None:
            self.context_photo = to_photo(ctx)
            self.context_view.config(image=self.context_photo, text="")
        else:
            self.context_photo = None
            self.context_view.config(image="", text="(no full frame on disk for this one)")
        g = self.suggestions(icon)
        self.guesses = [n for n, _ in g]
        self._buttons(self.guess_box, self.guesses, "F", dict(g))
        done = [n for n in self.answers if n.startswith("item_")]
        named_reads = sum(self.counts[n] for n in done if self.answers[n] not in (UNKNOWN,))
        self.status.config(
            text=f"{name}: read {self.counts[name]:,} times ({self.counts[name] / max(self.total_reads, 1):.1%} of "
                 f"unnamed column readings)\n{len(done)} of {self.n_templates} answered, covering "
                 f"{named_reads / max(self.total_reads, 1):.0%} of those readings   |   "
                 f"{len(self.queue) - self.pos} left")
        self.query.set("")
        self.refresh_results()
        self.entry.focus_set()

    def refresh_results(self):
        q = self.query.get().strip()
        self.results = fuzzy_search(q, self.names) if q else []
        self._buttons(self.result_box, self.results, "")

    def assign(self, label: str):
        if self.pos >= len(self.queue):
            return
        name = self.queue[self.pos]
        self.save(name, label)
        self.history.append(name)
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
        self.save(name, None)
        if name in self.queue:
            self.queue.remove(name)
        self.pos = max(0, min(self.pos - 1, len(self.queue)))
        self.queue.insert(self.pos, name)
        self.show()

    def quit(self):
        self.root.destroy()


# ── apply ─────────────────────────────────────────────────────────────────────

def apply() -> None:
    """
    Rename answered templates to their item name ("Name~N" when the name is taken), move
    "not an item" ones to items_seen_removed/, and rewrite bench_items in the local state
    files to match (not-an-item readings are dropped from the list; "can't tell" stays item_XXX).
    """
    aliases = load_game_data().item_aliases      # "2 Reforgers" -> "Reforger", "10 gold" -> None
    answers = {t: (aliases[l] or NONE) if l in aliases else l for t, l in load_answers().items()}
    taken = {p.stem for p in SEEN.glob("*.png")}
    rename: dict[str, str | None] = {}
    for tmpl, label in sorted(answers.items()):
        src = SEEN / f"{tmpl}.png"
        if not src.exists() or label == UNKNOWN:
            continue
        if label == NONE:
            REMOVED.mkdir(exist_ok=True)
            shutil.move(str(src), REMOVED / src.name)
            rename[tmpl] = None
            continue
        new, k = label, 2
        while new in taken:
            new, k = f"{label}~{k}", k + 1
        src.rename(SEEN / f"{new}.png")
        taken.discard(tmpl)
        taken.add(new)
        rename[tmpl] = label          # bench_items hold base names
    print(f"templates: {sum(v is not None for v in rename.values())} named, "
          f"{sum(v is None for v in rename.values())} moved to {REMOVED.name}/")
    rename.update(aliases)            # older readings may hold bundle names: fold or drop them too
    pat = re.compile('"(' + "|".join(map(re.escape, rename)) + ')"')
    changed_vods = set()
    for f in sorted(paths.STATES.glob("*/game_??.jsonl")):
        text = f.read_text(encoding="utf-8")
        if not pat.search(text):
            continue
        rows = []
        for line in text.splitlines():
            r = json.loads(line)
            if r.get("bench_items") and any(n in rename for n in r["bench_items"] if n):
                r["bench_items"] = [rename.get(n, n) if n in rename else n for n in r["bench_items"]
                                    if not (n in rename and rename[n] is None)]
            rows.append(r)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        changed_vods.add(f.parent.name)
    print(f"state files rewritten in {len(changed_vods)} VOD(s): {' '.join(sorted(changed_vods))}")
    print("next: clean_states.py, build_dataset.py and upload_states.py for those VODs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply()
        return
    Namer().root.mainloop()


if __name__ == "__main__":
    main()
