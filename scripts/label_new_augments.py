"""
Name the augments the detector has never been told about: one screen per
unnamed augment template, not per game.

An augment template that has no name (assets/templates/augment_names.json)
is an augment no labelled game taught the detector. Naming the template once
names it in every game it shows up in, now and in later videos. Only templates
that decide a game's augments on the player's own board are queued (the many
templates cut from other players' bars never matter) and that decide at
least one game nobody labelled, the ones deciding the most such games first.

Usage:
    python scripts/label_new_augments.py          # unnamed templates not labelled yet
    python scripts/label_new_augments.py --all    # also ones labelled here before

Run scripts/redo_augments.py on new videos first: the batch skips augment
reading, so their new augments have no template until then.

Each screen shows the template and up to 6 icons cut from different frames
and games where it decided a slot, the closest official icons, and a search
over every augment of the set ("+" versions folded in, GameData.augment_label).

Keys (all are buttons too):
    Enter       accept the first suggestion       F1-F8  pick a suggestion
    type, 1-8   search all augments and pick a result
    Tab         not an augment (a junk template)
    Esc         can't tell / skip for now         Ctrl+Z back one      Ctrl+Q quit

Saves data/processed/augment_template_labels.json {template: name | "_junk"}.
Then run scripts/name_augment_templates.py (it uses these names before the
per-game votes) and re-clean (scripts/clean_states.py) the videos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
import tkinter as tk
from collections import Counter, defaultdict
from tkinter import font as tkfont

import cv2
import numpy as np

from scripts.label_augments import ACCENT, AUG_DATA, BG, DIM, FG, AugmentLabeler, feats, official_icons
from scripts.label_common import load, merge_save, search_champions as fuzzy, to_photo
from src.state.game_data import GameData
from src.state.sequence import augment_name, game_augments
from src.utils import paths
from src.vision.arena import home_board_rows
from src.vision.augments import AugmentReader, expected_counts

TEMPLATE_LABELS = paths.REPO / "data" / "processed" / "augment_template_labels.json"
GAME_LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
JUNK = "_junk"
EXAMPLES = 6


def frame_dir(vod: str, game: str) -> Path | None:
    for root in (paths.FRAMES, paths.AUG_FRAMES):
        if (root / vod / game).exists():
            return root / vod / game
    return None


def candidates(skip: set[str]) -> list[dict]:
    """Unnamed templates that decide a slot of some game's augments, most unlabelled games first."""
    labelled = set(load(GAME_LABELS))
    found: dict[str, dict] = {}
    for states in sorted(paths.STATES.glob("*/game_??.jsonl")):
        vod, game = states.parent.name, states.stem
        rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                      key=lambda r: r["frame"])
        rows = [r for r in rows if r.get("in_game")]
        home = home_board_rows(rows)
        names, _, _ = game_augments(rows, home)
        for k, name in enumerate(names):
            if not name or not re.fullmatch(r"aug_\d+", name) or name in skip:
                continue
            c = found.setdefault(name, {"template": name, "games": [], "frames": [], "tiers": Counter()})
            c["games"].append((vod, game, k, f"{vod}/{game}" in labelled))
            for r, h in zip(rows, home):
                augs, tiers = r.get("augments") or [], r.get("augment_tiers") or []
                if h and k < len(augs) and augs[k] == name and not r.get("away_augments"):
                    c["frames"].append((vod, game, k, r))
                    if k < len(tiers):
                        c["tiers"][tiers[k]] += 1
    # Only templates that decide at least one unlabelled game: one whose games are all labelled
    # is unnamed because two augments share it (Pandora's Bench / Team Building), and one name
    # for it would be wrong for the other.
    out = [c for c in found.values() if any(not g[3] for g in c["games"])]
    out.sort(key=lambda c: (-sum(1 for g in c["games"] if not g[3]), -len(c["games"])))
    return out


class NewAugmentLabeler:
    def __init__(self, show_all: bool):
        self.reader = AugmentReader()
        self.official = []
        for stem, img in official_icons():
            name, _, tier = stem.rpartition("__")
            self.official.append((GameData.augment_label(name), tier, img, (feats(img), feats(img, bright=True))))
        aug_file = next(AUG_DATA.glob("*_augments.json"), None)
        names = json.loads(aug_file.read_text(encoding="utf-8")) if aug_file else [n for n, _, _, _ in self.official]
        self.names = sorted({GameData.augment_label(n) for n in names})
        self.labels = load(TEMPLATE_LABELS)
        print("Finding unnamed templates that decide games' augments ...")
        self.queue = candidates(set() if show_all else set(self.labels))
        print(f"{len(self.queue)} to name")
        self.i, self.history = 0, []
        self._build_window()
        self.show()

    # ── data ──────────────────────────────────────────────────────────────────

    def examples(self, c: dict) -> list[np.ndarray]:
        """Up to EXAMPLES icons of this template, from frames spread over its games."""
        by_game = defaultdict(list)
        for f in c["frames"]:
            by_game[(f[0], f[1])].append(f)
        picks, games = [], list(by_game.values())
        while len(picks) < EXAMPLES and any(games):
            for g in games:
                if g and len(picks) < EXAMPLES:
                    picks.append(g.pop(len(g) // 2))
        out = []
        for vod, game, k, r in picks:
            folder = frame_dir(vod, game)
            img = cv2.imread(str(folder / f"frame_{r['frame']:06d}.jpg")) if folder else None
            if img is None:
                continue
            icons = self.reader.icons(img, expected_counts(r.get("stage"), r.get("round_num")))
            if k < len(icons):
                out.append(icons[k])
        return out

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        r = self.root = tk.Tk()
        r.title("Name new augments")
        r.configure(bg=BG)
        self.big, self.small = tkfont.Font(size=13), tkfont.Font(size=10)
        self.status = tk.Label(r, bg=BG, fg=DIM, font=self.small, anchor="w", justify="left")
        self.status.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))
        actions = tk.Frame(r, bg=BG)
        actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))
        for text, cmd, colour in (
                ("Accept top suggestion  [Enter]", self.enter, "#2f6b45"),
                ("Not an augment (junk)  [Tab]", lambda: self.choose(JUNK), "#33353f"),
                ("Can't tell / skip  [Esc]", self.skip, "#33353f"),
                ("Back one  [Ctrl+Z]", self.back, "#33353f"),
                ("Quit  [Ctrl+Q]", lambda: r.destroy(), "#5a2f2f")):
            tk.Button(actions, text=text, command=cmd, font=self.big, bg=colour, fg=FG, relief="flat",
                      activebackground="#4a4d5a", activeforeground=FG, padx=10, pady=4).pack(side="left", padx=(0, 6))
        self.examples_row = tk.Frame(r, bg=BG)
        self.examples_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=6)
        self.info = tk.Label(r, bg=BG, fg=DIM, font=self.small, anchor="w", justify="left", wraplength=1100)
        self.info.grid(row=3, column=0, columnspan=2, sticky="we", padx=12)
        self.sugg_box = tk.Frame(r, bg=BG)
        self.sugg_box.grid(row=4, column=0, sticky="nw", padx=12, pady=6)
        right = tk.Frame(r, bg=BG)
        right.grid(row=4, column=1, sticky="nw", padx=6, pady=6)
        self.query = tk.StringVar()
        self.entry = tk.Entry(right, textvariable=self.query, font=self.big, width=30, bg="#2b2d35", fg=FG,
                              insertbackground=FG, relief="flat")
        self.entry.pack(anchor="w", ipady=4)
        tk.Label(right, text="type to search all augments, then click a result (or press 1-8)", bg=BG, fg=DIM,
                 font=self.small).pack(anchor="w")
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(right, bg=BG)
        self.result_box.pack(anchor="w", fill="x")
        r.bind("<Return>", lambda e: self.enter())
        r.bind("<Tab>", lambda e: (self.choose(JUNK), "break")[1])
        r.bind("<Escape>", lambda e: self.skip())
        r.bind("<Control-z>", lambda e: self.back())
        r.bind("<Control-q>", lambda e: r.destroy())
        for n in range(1, 9):
            r.bind(str(n), lambda e, k=n: (self.pick_result(k - 1), "break")[1])
            r.bind(f"<F{n}>", lambda e, k=n: self.pick_sugg(k - 1))

    def show(self):
        for box in (self.examples_row, self.sugg_box):
            for w in box.winfo_children():
                w.destroy()
        if self.i >= len(self.queue):
            self.status.configure(text="All done: every unnamed template that decides a game is labelled. "
                                       "Run scripts/name_augment_templates.py, then re-clean. Close the window.")
            self.info.configure(text="")
            self.sugg = []
            return
        c = self.queue[self.i]
        tmpl = self.reader.templates.get(c["template"])
        icons = ([tmpl] if tmpl is not None else []) + self.examples(c)
        self.photos = []
        for j, icon in enumerate(icons):
            ph = to_photo(cv2.resize(icon, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
            self.photos.append(ph)
            cell = tk.Frame(self.examples_row, bg=BG)
            cell.pack(side="left", padx=(0, 8))
            tk.Label(cell, image=ph, bg=BG).pack()
            tk.Label(cell, text="template" if (j == 0 and tmpl is not None) else "from a game", bg=BG, fg=DIM,
                     font=self.small).pack()
        games = c["games"]
        unl = sum(1 for g in games if not g[3])
        tiers = ", ".join(f"{t} {n}" for t, n in c["tiers"].most_common())
        listed = ", ".join(f"{v}/{g}" + ("" if lab else " (unlabelled)") for v, g, _, lab in games[:6])
        self.info.configure(text=f"Decides a slot in {len(games)} game(s), {unl} not labelled: {listed}"
                                 + (" ..." if len(games) > 6 else "") + f"   |   colour read as: {tiers or '-'}")
        self.status.configure(text=f"Template {self.i + 1} of {len(self.queue)}: {c['template']}   |   "
                                   f"named this session: {len(self.history)}")
        self.sugg = AugmentLabeler.suggestions(self, icons[1] if len(icons) > 1 else icons[0]) if icons else []
        tk.Label(self.sugg_box, text="Closest official icons", bg=BG, fg=DIM, font=self.small).pack(anchor="w")
        for k, (n, t, s) in enumerate(self.sugg):
            img = next(o for nn, tt, o, _ in self.official if nn == n and tt == t)
            ph = to_photo(cv2.resize(img, (48, 48)))
            b = tk.Button(self.sugg_box, text=f"F{k + 1}  {n}  ({t})", image=ph, compound="left", anchor="w",
                          font=self.big, bg="#2b2d35", fg=ACCENT if k == 0 else FG, relief="flat",
                          command=lambda n=n: self.choose(n))
            b.image = ph
            b.pack(fill="x", pady=1)
        self.query.set("")
        self.refresh_results()
        self.entry.focus_set()

    def refresh_results(self):
        self.results = fuzzy(self.query.get(), self.names)
        for w in self.result_box.winfo_children():
            w.destroy()
        for k, n in enumerate(self.results):
            tk.Button(self.result_box, text=f"{k + 1}  {n}", anchor="w", font=self.big, bg="#2b2d35", fg=FG,
                      relief="flat", command=lambda n=n: self.choose(n)).pack(fill="x", pady=1)

    # ── flow ──────────────────────────────────────────────────────────────────

    def enter(self):
        if self.query.get().strip():
            if self.results:
                self.choose(self.results[0])
        elif self.sugg:
            self.choose(self.sugg[0][0])

    def pick_sugg(self, k):
        if k < len(self.sugg):
            self.choose(self.sugg[k][0])

    def pick_result(self, k):
        if k < len(self.results):
            self.choose(self.results[k])

    def choose(self, name: str):
        if self.i >= len(self.queue):
            return
        tid = self.queue[self.i]["template"]
        merge_save(TEMPLATE_LABELS, {tid: name})
        self.labels[tid] = name
        self.history.append(tid)
        self.i += 1
        self.show()

    def skip(self):
        if self.i < len(self.queue):
            self.queue.append(self.queue.pop(self.i))
            self.show()

    def back(self):
        """Undo the last name given: its template comes back on screen."""
        if not self.history:
            return
        tid = self.history.pop()
        merge_save(TEMPLATE_LABELS, {tid: None})
        self.labels.pop(tid, None)
        j = next(k for k, c in enumerate(self.queue) if c["template"] == tid)   # always before self.i
        self.queue.insert(self.i - 1, self.queue.pop(j))
        self.i -= 1
        self.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Also templates labelled here before")
    args = parser.parse_args()
    app = NewAugmentLabeler(args.all)
    app.root.mainloop()


if __name__ == "__main__":
    main()
