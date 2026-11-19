"""
Label each game's three augments, one screen per game.

Usage:
    python scripts/label_augments.py              # games not labelled yet
    python scripts/label_augments.py --all        # every game, labelled ones too

A game has exactly three augments, picked at 2-1, 3-2 and 4-2 and kept to the
end, so one look at a late own-board frame labels the whole game. The screen
shows that frame's augment bar enlarged and, for each icon in turn, the five
closest official augment icons (assets/templates/augments_official/, with
their tier) plus what the reader made of it over the whole game.

Keys:
    Enter       accept the first suggestion for the highlighted icon
    F1-F8       pick a suggestion          1-8   pick a search result (type to search all augments)
    Tab         no augment in this slot (the game has fewer)
    Shift+Tab   an augment is here but blocked / can't be read (try Ctrl+N first)
    Ctrl+N      show another frame of this game (the bar is covered or unclear)
    Ctrl+Z      back one icon              Esc   skip the game     Ctrl+Q  quit

Saves data/processed/augment_labels.json {"<vod>/game_XX": {"augments": [...], "tiers": [...]}}
(read by scripts/clean_states.py), and each labelled icon to
~/tft_data/augment_crops/ (for naming the reader's templates later).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import pickle
from collections import Counter
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

from scripts.label_common import load, merge_save, search_champions as fuzzy, to_photo
from src.state.sequence import game_augments
from src.utils import paths
from src.vision.arena import home_board_rows
from src.vision.augments import AugmentReader, expected_counts
from src.vision.unit_bar import item_hue, item_vec

LABELS = paths.REPO / "data" / "processed" / "augment_labels.json"
OFFICIAL = paths.REPO / "assets" / "templates" / "augments_official"
AUG_DATA = paths.REPO / "assets" / "game_data"
CROPS = paths.LOCAL / "augment_crops"
NO_AUGMENT = "_none"
BLOCKED = "_unknown"       # an augment is there, but covered or unreadable: cleaning keeps the reader's guess
BG, FG, DIM, ACCENT = "#1e1f24", "#e8e8ea", "#8b8d98", "#f2c14e"


def feats(img, bright: bool = False, trim: int = 0):
    """Shape x colour features at 32 px; bright: stretch to full brightness (footage icons are dim)."""
    if trim:
        img = img[trim:-trim, trim:-trim]
    h, w = img.shape[:2]
    s = min(h, w)
    sq = cv2.resize(img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s], (32, 32),
                    interpolation=cv2.INTER_AREA)
    if bright:
        sq = np.clip(sq.astype(np.float32) * 255.0 / max(np.percentile(sq, 99), 1), 0, 255).astype(np.uint8)
    return item_vec(sq), item_hue(sq)


OFFICIAL_CACHE = paths.LOCAL / "cache" / "augments_official.pkl"


def official_icons() -> list[tuple[str, np.ndarray]]:
    """
    (file stem, image) of every official augment icon. Read from one local cache
    file: reading the 317 PNGs one by one through Google Drive took ~40 s.
    Rebuilt when the folder's file list changes.
    """
    files = sorted(OFFICIAL.glob("*.png"))
    stems = [p.stem for p in files]
    try:
        cached = pickle.loads(OFFICIAL_CACHE.read_bytes())
        if cached["stems"] == stems:
            return list(zip(stems, cached["images"]))
    except (OSError, KeyError, pickle.UnpicklingError, EOFError):
        pass
    images = [cv2.imread(str(p)) for p in files]
    OFFICIAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OFFICIAL_CACHE.write_bytes(pickle.dumps({"stems": stems, "images": images}))
    return [(s, im) for s, im in zip(stems, images) if im is not None]


class AugmentLabeler:
    def __init__(self, show_all: bool):
        self.reader = AugmentReader()
        self.official = []
        for stem, img in official_icons():
            name, _, tier = stem.rpartition("__")
            self.official.append((name, tier, img, (feats(img), feats(img, bright=True))))
        aug_file = next(AUG_DATA.glob("*_augments.json"), None)
        self.names = sorted(json.loads(aug_file.read_text(encoding="utf-8"))) if aug_file else \
            sorted({n for n, _, _, _ in self.official})
        labels = load(LABELS)
        self.games = []
        for states in sorted(paths.STATES.glob("*/game_*.jsonl")):
            if states.name.endswith(".clean.jsonl"):
                continue
            key = f"{states.parent.name}/{states.stem}"
            if show_all or key not in labels:
                self.games.append((key, states))
        self.g = 0
        self._build_window()
        self.load_game()

    # ── data ──────────────────────────────────────────────────────────────────

    def load_game(self):
        while self.g < len(self.games):
            key, states = self.games[self.g]
            rows = sorted((json.loads(l) for l in states.read_text(encoding="utf-8").splitlines() if l),
                          key=lambda r: r["frame"])
            rows = [r for r in rows if r.get("in_game")]
            home = home_board_rows(rows)
            names, tiers, picked = game_augments(rows, home)
            # Candidate frames: own board, after the last pick (4-2), as many icons as the bar
            # showed there; late first. Counts, not names, so this works before templates exist.
            late = [r for r, h in zip(rows, home)
                    if h and r.get("augments") and (r.get("stage") or 0, r.get("round_num") or 0) > (4, 2)]
            # the most icons seen on enough frames (a count on 1-3 frames is a misread)
            seen = Counter(len(r["augments"]) for r in late)
            most = max((n for n, k in seen.items() if k >= max(5, 0.05 * len(late))), default=0)
            cands = [r for r in late if len(r["augments"]) == most]
            if not cands:   # game over before 4-2: the old rule (after the reader's last pick)
                after = picked[-1] if picked else 0
                cands = [r for i, (r, h) in enumerate(zip(rows, home))
                         if h and i >= after and len(r.get("augments") or []) == max(1, len(names))]
            cands = cands[::-1][::max(1, len(cands) // 12)] or [r for r, h in zip(rows, home) if h][-12:]
            frame_dir = paths.FRAMES / key.split("/")[0] / states.stem
            if not frame_dir.exists():     # full frames deleted: the augment-only copies (slim_frames.py)
                frame_dir = paths.AUG_FRAMES / key.split("/")[0] / states.stem
            # with each frame, the icon counts its stage allows (so each icon is cut on its own)
            self.cands = [(frame_dir / f"frame_{r['frame']:06d}.jpg", expected_counts(r.get("stage"), r.get("round_num")))
                          for r in cands]
            self.cands = [c for c in self.cands if c[0].exists()]
            if self.cands:
                self.key, self.read_names, self.read_tiers = key, names, tiers
                self.c = 0
                self.slot, self.chosen = 0, []
                self.show_frame()
                return
            self.g += 1
        self.status.configure(text="All games done. Close the window.")

    def show_frame(self):
        path, allowed = self.cands[self.c]
        self.frame = cv2.imread(str(path))
        self.icons = self.reader.icons(self.frame, allowed)
        x, y, w, h = self.reader.region
        pad = 20 + self.reader.SHIFT
        bar = self.frame[y - 6:y + h + 6, x - pad:x + w + pad]
        self.bar_photo = to_photo(cv2.resize(bar, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
        self.bar_view.configure(image=self.bar_photo)
        ctx = cv2.resize(self.frame, (640, 360), interpolation=cv2.INTER_AREA)
        self.ctx_photo = to_photo(ctx)
        self.ctx_view.configure(image=self.ctx_photo)
        self.show_slot()

    def suggestions(self, icon, per_method: int = 4):
        """
        Closest official icons by two measures - as is, and both brightness-
        stretched (footage icons are dim) - the top few of each, interleaved.
        Each finds icons the other misses; on 7 icons checked by eye the
        right one was in the combined list far more often than in either.
        """
        lists = []
        for m, f in enumerate((feats(icon), feats(icon, bright=True, trim=2))):
            lists.append(sorted(((float(f[0] @ o[m][0]) * float(f[1] @ o[m][1]), n, t)
                                 for n, t, _, o in self.official), reverse=True)[:per_method * 2])
        out, seen = [], set()
        for k in range(per_method * 2):
            for lst in lists:
                if k < len(lst) and (lst[k][1], lst[k][2]) not in seen and len(out) < per_method * 2:
                    seen.add((lst[k][1], lst[k][2]))
                    out.append((lst[k][1], lst[k][2], lst[k][0]))
        return out

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        r = self.root = tk.Tk()
        r.title("Label augments")
        r.configure(bg=BG)
        self.big, self.small = tkfont.Font(size=13), tkfont.Font(size=10)
        self.status = tk.Label(r, bg=BG, fg=DIM, font=self.small, anchor="w", justify="left")
        self.status.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))
        # Every shortcut as a button, at the top so it is always on screen.
        actions = tk.Frame(r, bg=BG)
        actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))
        for text, cmd, colour in (
                ("Accept top suggestion  [Enter]", self.enter, "#2f6b45"),
                ("No augment here  [Tab]", lambda: self.choose(NO_AUGMENT, None), "#33353f"),
                ("Blocked / can't read  [Shift+Tab]", lambda: self.choose(BLOCKED, None), "#33353f"),
                ("Another frame  [Ctrl+N]", self.next_frame, "#33353f"),
                ("Back one icon  [Ctrl+Z]", self.back, "#33353f"),
                ("Skip game  [Esc]", self.skip, "#33353f"),
                ("Quit  [Ctrl+Q]", lambda: r.destroy(), "#5a2f2f")):
            tk.Button(actions, text=text, command=cmd, font=self.big, bg=colour, fg=FG, relief="flat",
                      activebackground="#4a4d5a", activeforeground=FG, padx=10, pady=4).pack(side="left", padx=(0, 6))
        self.bar_view = tk.Label(r, bg=BG)
        self.bar_view.grid(row=2, column=0, columnspan=2, padx=12, sticky="w")
        self.icon_view = tk.Label(r, bg=BG)
        self.icon_view.grid(row=3, column=0, rowspan=3, padx=12, pady=8, sticky="n")
        self.sugg_box = tk.Frame(r, bg=BG)
        self.sugg_box.grid(row=3, column=1, sticky="nw", padx=6)
        q = tk.Frame(r, bg=BG)
        q.grid(row=4, column=1, sticky="w", padx=6, pady=4)
        self.query = tk.StringVar()
        self.entry = tk.Entry(q, textvariable=self.query, font=self.big, width=28, bg="#2b2d35", fg=FG,
                              insertbackground=FG, relief="flat")
        self.entry.pack(side="left", ipady=4)
        tk.Label(q, text="  type to search all augments, then click a result (or press 1-8)", bg=BG, fg=DIM,
                 font=self.small).pack(side="left")
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(r, bg=BG)
        self.result_box.grid(row=5, column=1, sticky="nw", padx=6)
        self.ctx_view = tk.Label(r, bg=BG)
        self.ctx_view.grid(row=6, column=0, columnspan=2, padx=12, pady=8, sticky="w")
        r.bind("<Return>", lambda e: self.enter())
        r.bind("<Tab>", lambda e: (self.choose(NO_AUGMENT, None), "break")[1])
        r.bind("<Shift-Tab>", lambda e: (self.choose(BLOCKED, None), "break")[1])
        try:
            r.bind("<ISO_Left_Tab>", lambda e: (self.choose(BLOCKED, None), "break")[1])
        except tk.TclError:
            pass
        r.bind("<Control-n>", lambda e: self.next_frame())
        r.bind("<Control-z>", lambda e: self.back())
        r.bind("<Escape>", lambda e: self.skip())
        r.bind("<Control-q>", lambda e: r.destroy())
        for n in range(1, 9):
            r.bind(str(n), lambda e, k=n: (self.pick_result(k - 1), "break")[1])
        for n in range(1, 9):
            r.bind(f"<F{n}>", lambda e, k=n: self.pick_sugg(k - 1))

    @property
    def n_slots(self) -> int:
        """Icons to label in this game: as many as the frame's bar shows (up to 4), at least 3."""
        return max(3, len(self.icons))

    def show_slot(self):
        if self.slot >= self.n_slots:
            return self.save_game()
        icon = self.icons[self.slot] if self.slot < len(self.icons) else None
        if icon is not None:
            self.icon_photo = to_photo(cv2.resize(icon, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
            self.icon_view.configure(image=self.icon_photo)
            self.sugg = self.suggestions(icon)
        else:
            self.icon_view.configure(image="")
            self.sugg = []
        for c in self.sugg_box.winfo_children():
            c.destroy()
        read = self.read_names[self.slot] if self.slot < len(self.read_names) else None
        tk.Label(self.sugg_box, text=f"Icon {self.slot + 1} of {self.n_slots}"
                 + ("" if icon is not None else "  - no icon on this frame (Tab if the game has fewer, Ctrl+N for another frame)")
                 + (f"   reader over the game: {read}" if read else ""), bg=BG, fg=DIM, font=self.small).pack(anchor="w")
        for i, (n, t, s) in enumerate(self.sugg):
            img = next(o for nn, tt, o, _ in self.official if nn == n and tt == t)
            ph = to_photo(cv2.resize(img, (48, 48)))
            b = tk.Button(self.sugg_box, text=f"F{i + 1}  {n}  ({t}, {s:.2f})", image=ph, compound="left",
                          anchor="w", font=self.big, bg="#2b2d35", fg=ACCENT if i == 0 else FG, relief="flat",
                          command=lambda n=n, t=t: self.choose(n, t))
            b.image = ph
            b.pack(fill="x", pady=1)
        self.query.set("")
        self.refresh_results()
        self.status.configure(text=f"Game {self.g + 1} of {len(self.games)}: {self.key}   |   frame {self.c + 1} of "
                                   f"{len(self.cands)} (Ctrl+N for another)   |   chosen so far: {self.chosen}")
        self.entry.focus_set()

    def refresh_results(self):
        self.results = fuzzy(self.query.get(), self.names)
        for c in self.result_box.winfo_children():
            c.destroy()
        for i, n in enumerate(self.results):
            tk.Button(self.result_box, text=f"{i + 1}  {n}", anchor="w", font=self.big, bg="#2b2d35", fg=FG,
                      relief="flat", command=lambda n=n: self.choose(n, None)).pack(fill="x", pady=1)

    # ── flow ──────────────────────────────────────────────────────────────────

    def enter(self):
        if self.query.get().strip():
            if self.results:
                self.choose(self.results[0], None)
        elif self.sugg:
            self.choose(self.sugg[0][0], self.sugg[0][1])

    def pick_sugg(self, i):
        if i < len(self.sugg):
            self.choose(self.sugg[i][0], self.sugg[i][1])

    def pick_result(self, i):
        if i < len(self.results):
            self.choose(self.results[i], None)

    def choose(self, name, tier):
        if name not in (NO_AUGMENT, BLOCKED) and tier is None and self.slot < len(self.icons):
            tier = self.reader.tier(self.icons[self.slot])        # typed name: tier from the icon colour
        self.chosen.append((name, tier))
        self.slot += 1
        self.show_slot()

    def back(self):
        if self.chosen:
            self.chosen.pop()
            self.slot -= 1
            self.show_slot()

    def next_frame(self):
        """Another frame of the same game; icons already picked are kept (carry on where you were)."""
        self.c = (self.c + 1) % len(self.cands)
        self.show_frame()

    def save_game(self):
        picked = [(n, t) for n, t in self.chosen if n != NO_AUGMENT]
        merge_save(LABELS, {self.key: {"augments": [n for n, _ in picked], "tiers": [t for _, t in picked],
                                       "frame": self.cands[self.c][0].name}})
        CROPS.mkdir(parents=True, exist_ok=True)
        for k, icon in enumerate(self.icons[:len(picked)]):
            cv2.imwrite(str(CROPS / f"{self.key.replace('/', '__')}__a{k}.png"), icon)
        self.g += 1
        self.load_game()

    def skip(self):
        self.g += 1
        self.load_game()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Every game, labelled ones too")
    args = parser.parse_args()
    app = AugmentLabeler(args.all)
    app.root.mainloop()


if __name__ == "__main__":
    main()
