"""
Check cleaned states against the full frame, by eye (Hamza's checker, step 5b).

Usage:
    python scripts/check_states.py                    # random in-game frames of the VODs with full frames
    python scripts/check_states.py --planning         # planning frames only (where units are read)
    python scripts/check_states.py --vods knB6dE8Rbgg --n 100
    python scripts/check_states.py --report           # accuracy per field from the checks so far

The frame is shown on the left (cleaned board and bench units drawn on their
hexes and slots; O toggles that; a 3x zoom of the spot under the mouse below it), every cleaned
field on the right as chips. Everything unmarked counts as right when you go
on. Click a chip: wrong (red, a box opens for what it should be, optional);
again: can't tell (yellow, left out of the figures); again: back to right.
Units have one chip each for champion, star and items, and "not there" for a
unit that isn't on that spot; "missed" boxes take units or items the state
left out. Next to each unit, in grey, what that single frame read at that
spot (so a mistake can be put on the reading or on the cleaning).

Keys: Enter = next (saves this frame), Ctrl+Backspace = back, O = overlay, D = dark / light
(default follows Windows; --dark / --light), Esc = quit (the frame on screen is not saved
unless Enter was pressed on it). The window sizes itself to the screen; hovering a unit's card
outlines its label on the frame, and a unit with a field marked wrong gets a red label.

Checks go to data/processed/state_checks.json, one entry per vod/game/frame:
{"shown": [field keys], "wrong": {key: {"value", "note"}}, "unsure": [keys],
"missed": {"board": text, ...}, "checked": time}. --report turns them into
per-field accuracy (unsure fields left out; "missed" counted separately).
Needs the full frames (~/tft_frames_temp), so it only works before step 6.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time
import tkinter as tk
from collections import Counter, defaultdict

import cv2
import numpy as np

from scripts.label_common import load, merge_save, to_photo
from src.utils import paths

CHECKS = paths.REPO / "data" / "processed" / "state_checks.json"
ZOOM = 3                            # the zoom box shows the frame under the mouse at 3x
OK, WRONG, UNSURE = "ok", "wrong", "unsure"
MISSED = ("board", "bench", "item column")


# ── fields (no Tk: tested on their own) ─────────────────────────────────────

def _num(v):
    return "-" if v is None else (str(int(v)) if float(v).is_integer() else f"{v:g}")


def _items(items) -> str:
    return ", ".join(items) if items else "no items"


def fields_of(row: dict) -> list[tuple[str, str, str, str]]:
    """
    Every cleaned field of a row as (key, group, label, value shown).
    One chip per key; the group is what --report scores together.
    """
    out = [
        ("stage", "stage", "Stage", f"{_num(row.get('stage'))}-{_num(row.get('round_num'))}"),
        ("gold", "gold", "Gold", _num(row.get("gold"))),
        ("hp", "hp", "HP", _num(row.get("hp"))),
        ("level", "level", "Level", _num(row.get("level"))),
        ("xp", "xp", "XP", f"{_num(row.get('xp_current'))}/{_num(row.get('xp_needed'))}"),
        ("streak", "streak", "Streak", _num(row.get("streak"))),
        ("planning", "planning", "Planning", "planning" if row.get("planning") else "not planning (combat etc.)"),
        ("home_board", "home board", "Own board", "own board" if row.get("home_board") else "someone else's board"),
    ]
    for i, name in enumerate(row.get("shop_slots") or []):
        out.append((f"shop.{i}", "shop", f"Shop {i + 1}", name or "empty / not read"))
    for i, name in enumerate(row.get("bench_items") or []):
        out.append((f"item.{i}", "item column", f"Item {i + 1}", name or "unknown"))
    tiers = row.get("augment_tiers") or []
    for i, name in enumerate(row.get("augments") or []):
        tier = tiers[i] if i < len(tiers) and tiers[i] not in (None, "unknown") else ""
        out.append((f"augment.{i}", "augments", f"Augment {i + 1}", f"{name or 'unknown'}{f' ({tier})' if tier else ''}"))
    lobby = row.get("other_hp") or []
    if lobby:
        out.append(("lobby_hp", "lobby hp", "Lobby HP", " ".join(_num(v) for v in lobby)))
    for u in row.get("board") or []:
        at = f"h{u['hex'][0]}{u['hex'][1]}"
        out += _unit_fields("board", at, u)
    for u in row.get("bench") or []:
        out += _unit_fields("bench", f"s{u['slot']}", u)
    return out


def _unit_fields(where: str, at: str, u: dict) -> list[tuple[str, str, str, str]]:
    star = u.get("star")
    flags = ",".join(k for k in ("carried", "filled", "loose") if u.get(k))
    label = f"{where} {at}" + (f" ({flags})" if flags else "")
    return [
        (f"{where}.{at}.name", "unit champion", label, u.get("name") or "?"),
        (f"{where}.{at}.star", "unit star", label, f"{star} star" if star else "star ?"),
        (f"{where}.{at}.items", "unit items", label, _items(u.get("items"))),
        (f"{where}.{at}.there", "unit there", label, "not there"),
    ]


def group_of(key: str) -> str:
    """The --report group of a field key (as fields_of names them)."""
    head = key.split(".")[0]
    if head in ("board", "bench"):
        return {"name": "unit champion", "star": "unit star", "items": "unit items",
                "there": "unit there"}[key.split(".")[-1]]
    return {"shop": "shop", "item": "item column", "augment": "augments", "lobby_hp": "lobby hp",
            "home_board": "home board"}.get(head, head)


def report(checks: dict) -> list[tuple[str, int, int]]:
    """(group, fields judged, wrong) per group: shown minus unsure; "unit there" only counts marks."""
    judged, wrong = Counter(), Counter()
    for entry in checks.values():
        unsure = set(entry.get("unsure", []))
        for key in entry.get("shown", []):
            if key in unsure:
                continue
            judged[group_of(key)] += 1
            wrong[group_of(key)] += key in entry.get("wrong", {})
    return [(g, judged[g], wrong[g]) for g in sorted(judged, key=lambda g: -judged[g])]


def sample(rows_by_game: dict, n: int, done: set, planning_only: bool, seed: int = 0) -> list[tuple]:
    """Up to n (vod, game, frame) not checked yet, spread evenly over the games."""
    rng = random.Random(seed)
    pools = []
    for (vod, game), rows in sorted(rows_by_game.items()):
        keys = [(vod, game, r["frame"]) for r in rows
                if f"{vod}/{game}/{r['frame']}" not in done and (r.get("planning") or not planning_only)]
        rng.shuffle(keys)
        if keys:
            pools.append(keys)
    out = []
    while pools and len(out) < n:
        for pool in list(pools):
            if pool:
                out.append(pool.pop())
            else:
                pools.remove(pool)
            if len(out) >= n:
                break
    rng.shuffle(out)
    return out


def raw_at(raw_row: dict | None, where: str, at: str) -> str:
    """What the single frame read at a position (units_raw), for the grey hint."""
    if not raw_row or raw_row.get("units_raw") is None:
        return "frame not read"
    for u in raw_row["units_raw"]:
        if u.get("at") == at:
            if u.get("covered"):
                return "frame: under a tooltip"
            return f"frame: {u.get('name')} {u.get('p', 0):.2f}" + (" loose" if u.get("loose") else "")
    return "frame: nothing here"


# ── the app ─────────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg": "#1e1f22", "panel": "#26282c", "card": "#2f3136", "line": "#3a3d43",
        "text": "#e8e8e8", "muted": "#a0a4aa", "faint": "#71767d", "entry": "#1a1b1e",
        OK: ("#2c4a36", "#d6f2dc"), WRONG: ("#8a2f2f", "#ffe1e1"), UNSURE: ("#6e5a17", "#fff0c0"),
        "off": ("#383a40", "#a0a4aa"),
    },
    "light": {
        "bg": "#eef0f3", "panel": "#ffffff", "card": "#f6f7f9", "line": "#dcdfe4",
        "text": "#1f2328", "muted": "#5c6168", "faint": "#8a9097", "entry": "#ffffff",
        OK: ("#d9eedb", "#1d4424"), WRONG: ("#f3bcbc", "#6a1111"), UNSURE: ("#f8e3a0", "#5a4300"),
        "off": ("#e6e8eb", "#5c6168"),
    },
}
MARK = {OK: "", WRONG: "✕ ", UNSURE: "? "}          # state shown by a sign too, not by colour alone


def windows_dark() -> bool:
    """Windows' app theme (dark unless 'AppsUseLightTheme' is 1)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def _dpi_aware():
    """Real pixels on a scaled display (125% etc.): sharp frame, and the true screen size."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def _work_area(root) -> tuple[int, int]:
    """Screen size minus the taskbar, in pixels."""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(r), 0)   # SPI_GETWORKAREA
        return r.right - r.left, r.bottom - r.top
    except (AttributeError, OSError):
        return root.winfo_screenwidth(), root.winfo_screenheight() - 60


class Checker:
    def __init__(self, root: tk.Tk, queue: list[tuple], clean: dict, raw: dict, dark: bool = True):
        from src.vision.bench import BenchSlots
        from src.vision.board import BoardHexes
        self.board, self.bench = BoardHexes(), BenchSlots()
        self.root, self.queue, self.clean, self.raw = root, queue, clean, raw
        self.pos = 0
        self.changes: dict = {}
        self.overlay = True
        self.theme = "dark" if dark else "light"
        self.states: dict[str, str] = {}
        self.chips: dict[str, tk.Label] = {}
        self.notes: dict[str, tk.Entry] = {}
        self.missed: dict[str, tk.Entry] = {}
        self.hover = None                     # (where, at) of the unit card under the mouse
        self.img = self.zoom_img = None

        # sizes from the screen: the field panel gets a fixed share, the frame the rest
        self.s = root.winfo_fpixels("1i") / 96                     # 1.25 at 125%
        w, h = _work_area(root)
        self.panel_w = max(int(470 * self.s), int(w * 0.30))
        self.zoom_wh = (int(330 * self.s), int(205 * self.s))
        chrome = int(40 * self.s)                                   # title bar, margins
        self.scale = min((w - self.panel_w - int(36 * self.s)) / 1920,
                         (h - chrome - self.zoom_wh[1] - int(24 * self.s)) / 1080)
        root.title("State checker")
        root.geometry(f"{w}x{h - chrome}+0+0")
        try:
            root.state("zoomed")
        except tk.TclError:
            pass
        root.bind("<Return>", lambda e: self.next())
        root.bind("<Control-BackSpace>", lambda e: None if isinstance(e.widget, tk.Entry) else self.back())
        root.bind("<Escape>", lambda e: self.quit())
        for k in "oO":
            root.bind(f"<Key-{k}>", self.toggle_overlay)
        for k in "dD":
            root.bind(f"<Key-{k}>", self.toggle_theme)
        root.bind_all("<MouseWheel>", self.wheel)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.build()
        self.show()

    # ── layout ──
    def px(self, n) -> int:
        return int(n * self.s)

    def build(self):
        t = THEMES[self.theme]
        self.t = t
        root = self.root
        root.configure(bg=t["bg"])
        self._title_bar()
        pad = self.px(10)

        left = tk.Frame(root, bg=t["bg"])
        left.grid(row=0, column=0, sticky="nw", padx=(pad, pad // 2), pady=pad)
        self.canvas = tk.Label(left, bg=t["bg"], bd=0)
        self.canvas.grid(row=0, column=0, columnspan=2, sticky="nw")
        self.canvas.bind("<Motion>", self.zoom_at)

        zw, zh = self.zoom_wh
        zbox = tk.Frame(left, bg=t["line"], width=zw + 2, height=zh + 2)
        zbox.grid(row=1, column=0, sticky="nw", pady=(pad, 0))
        zbox.grid_propagate(False)
        self.zoom = tk.Label(zbox, bg="#000000", bd=0, text="zoom: move the mouse over the frame",
                             fg=t["faint"], font=("Segoe UI", 9))
        self.zoom.place(x=1, y=1, width=zw, height=zh)

        info = tk.Frame(left, bg=t["bg"])
        info.grid(row=1, column=1, sticky="nw", padx=(pad * 2, 0), pady=(pad, 0))
        self.status = tk.Label(info, font=("Segoe UI Semibold", 11), bg=t["bg"], fg=t["text"],
                               anchor="w", justify="left")
        self.status.pack(anchor="w")
        self.substatus = tk.Label(info, font=("Segoe UI", 9), bg=t["bg"], fg=t["muted"], anchor="w")
        self.substatus.pack(anchor="w")
        legend = tk.Frame(info, bg=t["bg"])
        legend.pack(anchor="w", pady=(self.px(8), self.px(4)))
        for state, text in ((OK, "right"), (WRONG, "wrong"), (UNSURE, "can't tell")):
            bg, fg = t[state]
            tk.Label(legend, text=MARK[state] + text, bg=bg, fg=fg, font=("Segoe UI", 9),
                     padx=self.px(7), pady=self.px(1)).pack(side="left", padx=(0, self.px(5)))
        tk.Label(legend, text="click a chip to cycle", bg=t["bg"], fg=t["faint"],
                 font=("Segoe UI", 9)).pack(side="left", padx=self.px(4))
        keys = ("Enter  next (unmarked = right)      Ctrl+Backspace  back\n"
                "O  unit labels on the frame      D  dark / light      Esc  quit (frame on screen not saved)")
        tk.Label(info, text=keys, font=("Segoe UI", 9), bg=t["bg"], fg=t["muted"], justify="left").pack(anchor="w")
        self.score = tk.Label(info, font=("Consolas", 9), bg=t["bg"], fg=t["muted"], anchor="w", justify="left")
        self.score.pack(anchor="w", pady=(self.px(8), 0))

        side = tk.Frame(root, bg=t["panel"], width=self.panel_w)
        side.grid(row=0, column=1, sticky="nsew", pady=pad, padx=(pad // 2, pad))
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self.side = side
        self.list_canvas = tk.Canvas(side, bg=t["panel"], highlightthickness=0, bd=0)
        bar = tk.Scrollbar(side, orient="vertical", command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.list_canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.list_canvas, bg=t["panel"])
        win = self.list_canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")))
        self.list_canvas.bind("<Configure>", lambda e: self.list_canvas.itemconfigure(win, width=e.width))

    def _title_bar(self):
        """Dark or light Windows title bar to match (Windows 10 20H1+ / 11; ignored elsewhere)."""
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            on = ctypes.c_int(1 if self.theme == "dark" else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
        except (AttributeError, OSError):
            pass

    def wheel(self, e):
        """The mouse wheel scrolls the field list when the pointer is over it."""
        if str(e.widget).startswith(str(self.side)):
            self.list_canvas.yview_scroll(-e.delta // 120, "units")

    # ── one frame ──
    def key(self):
        vod, game, frame = self.queue[self.pos]
        return f"{vod}/{game}/{frame}"

    def show(self, old: dict | None = None):
        if self.pos >= len(self.queue):
            self.status.config(text="All done.")
            self.substatus.config(text="")
            return
        vod, game, frame = self.queue[self.pos]
        self.row = self.clean[(vod, game)][frame]
        self.raw_row = self.raw.get((vod, game), {}).get(frame)
        self.frame = cv2.imread(str(paths.FRAMES / vod / game / f"frame_{frame:06d}.jpg"))
        if self.frame is None:
            self.frame = np.zeros((1080, 1920, 3), np.uint8)
        if old is None:
            old = self.changes.get(self.key()) or load(CHECKS).get(self.key()) or {}
        self.fields = fields_of(self.row)
        self.states = {k: OK for k, _, _, _ in self.fields}
        self.states.update({k: WRONG for k in old.get("wrong", {}) if k in self.states})
        self.states.update({k: UNSURE for k in old.get("unsure", []) if k in self.states})
        self.hover = None
        self.draw_frame()
        self.draw_fields(old)
        self.status.config(text=f"Frame {self.pos + 1} of {len(self.queue)}")
        self.substatus.config(text=f"{vod}  ·  {game}  ·  frame {frame} ({frame / 2 / 60:.1f} min)  ·  "
                                   f"augments from {self.row.get('augment_source', '?')}")
        self.draw_score()

    # ── the frame, with each cleaned unit's name on its hex / slot ──
    def _unit_spots(self):
        for u in self.row.get("board") or []:
            cx, cy = self.board.center(*u["hex"])
            yield "board", f"h{u['hex'][0]}{u['hex'][1]}", u, cx, cy, cy + 25
        for u in self.row.get("bench") or []:
            cx = int(self.bench.center_x(u["slot"]))
            top = self.bench.region(u["slot"])[1]
            yield "bench", f"s{u['slot']}", u, cx, top + 70, top + 120

    def draw_frame(self):
        small = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.overlay:
            for where, at, u, cx, cy, ty in self._unit_spots():
                wrong = any(self.states.get(f"{where}.{at}.{f}") == WRONG for f in ("name", "star", "items", "there"))
                self._tag(small, cx * self.scale, ty * self.scale, u, wrong, self.hover == (where, at))
        self.img = to_photo(small)
        self.canvas.config(image=self.img)

    @staticmethod
    def _tag(img, cx, y, u, wrong, hot):
        """Two short lines (name; star and item count), kept inside the picture."""
        name = u.get("name") or "?"
        name = name if len(name) <= 12 else name[:11] + "."
        sub = f"{u.get('star') or '?'}*" + (f"  {len(u['items'])} it" if u.get("items") else "")
        font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.42
        (w1, h1), _ = cv2.getTextSize(name, font, fs, 1)
        (w2, _), _ = cv2.getTextSize(sub, font, fs * 0.9, 1)
        w, h = max(w1, w2) + 8, 2 * h1 + 12
        H, W = img.shape[:2]
        x0 = int(min(max(2, cx - w / 2), W - w - 2))
        y0 = int(min(max(2, y - h / 2), H - h - 2))
        box = img[y0:y0 + h, x0:x0 + w]
        fill = np.array((40, 40, 170) if wrong else (20, 20, 20), np.float32)
        box[:] = (box * 0.25 + fill * 0.75).astype(np.uint8)
        if hot:
            cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + w + 1, y0 + h + 1), (0, 230, 255), 2)
        colour = (80, 210, 255) if u.get("carried") or u.get("filled") else (255, 255, 255)
        cv2.putText(img, name, (x0 + 4, y0 + h1 + 4), font, fs, colour, 1, cv2.LINE_AA)
        cv2.putText(img, sub, (x0 + 4, y0 + 2 * h1 + 9), font, fs * 0.9, (190, 190, 190), 1, cv2.LINE_AA)

    def zoom_at(self, event):
        zw, zh = self.zoom_wh
        pw, ph = int(zw / ZOOM), int(zh / ZOOM)                   # patch of the full frame, 3x
        x, y = int(event.x / self.scale), int(event.y / self.scale)
        x0 = min(max(0, x - pw // 2), 1920 - pw)
        y0 = min(max(0, y - ph // 2), 1080 - ph)
        patch = cv2.resize(self.frame[y0:y0 + ph, x0:x0 + pw], (zw, zh), interpolation=cv2.INTER_NEAREST)
        self.zoom_img = to_photo(patch)
        self.zoom.config(image=self.zoom_img, text="")

    # ── the field panel ──
    def _section(self, title: str, count: int | None = None):
        t = self.t
        head = tk.Frame(self.inner, bg=t["panel"])
        head.pack(fill="x", padx=self.px(12), pady=(self.px(14), self.px(4)))
        tk.Label(head, text=title.upper(), font=("Segoe UI Semibold", 9), bg=t["panel"],
                 fg=t["muted"]).pack(side="left")
        if count is not None:
            tk.Label(head, text=str(count), font=("Segoe UI", 9), bg=t["panel"], fg=t["faint"]).pack(side="left", padx=6)
        tk.Frame(head, bg=t["line"], height=1).pack(side="left", fill="x", expand=True, padx=(self.px(8), 0), pady=(2, 0))
        body = tk.Frame(self.inner, bg=t["panel"])
        body.pack(fill="x", padx=self.px(12))
        return body

    def _chip(self, parent, key, value, bg_parent, wrap):
        t = self.t
        state = self.states[key]
        there = key.endswith(".there")
        bg, fg = t["off"] if there and state == OK else t[state]
        chip = tk.Label(parent, text=self._chip_text(key, value), bg=bg, fg=fg, font=("Segoe UI", 10),
                        padx=self.px(8), pady=self.px(3), anchor="w", justify="left", wraplength=wrap,
                        cursor="hand2")
        chip.bind("<Button-1>", lambda e, k=key: self.cycle(k))
        self.chips[key] = chip
        self.values[key] = value
        note = tk.Entry(parent, font=("Segoe UI", 9), bg=t["entry"], fg=t["text"], insertbackground=t["text"],
                        relief="flat", highlightthickness=1, highlightbackground=t[WRONG][0],
                        highlightcolor=t[WRONG][0])
        self.notes[key] = note
        return chip, note

    def _chip_text(self, key, value):
        return MARK[self.states[key]] + value

    def _grid(self, body, cells, ncols, captions=True):
        """Tiles in ncols equal columns: caption, chip, and the 'should be' box under it when wrong."""
        t = self.t
        col_w = (self.panel_w - self.px(60)) // ncols
        for c in range(ncols):
            body.grid_columnconfigure(c, weight=1, uniform="tiles")
        for i, (key, caption, value) in enumerate(cells):
            cell = tk.Frame(body, bg=t["panel"])
            cell.grid(row=i // ncols, column=i % ncols, sticky="new", padx=(0, self.px(6)), pady=self.px(3))
            if captions and caption:
                tk.Label(cell, text=caption, font=("Segoe UI", 8), bg=t["panel"], fg=t["faint"]).pack(anchor="w")
            chip, note = self._chip(cell, key, value, t["panel"], col_w - self.px(20))
            chip.pack(fill="x")
            self._note_slot[key] = lambda n=note: n.pack(fill="x", pady=(self.px(2), 0))

    def _unit_card(self, body, where, at, label, cells):
        t = self.t
        card = tk.Frame(body, bg=t["card"], highlightthickness=1, highlightbackground=t["line"])
        card.pack(fill="x", pady=self.px(3))
        top = tk.Frame(card, bg=t["card"])
        top.pack(fill="x", padx=self.px(8), pady=(self.px(5), 0))
        pos, _, flags = label.partition(" (")
        tk.Label(top, text=at, font=("Segoe UI Semibold", 10), bg=t["card"], fg=t["text"]).pack(side="left")
        if flags:
            tk.Label(top, text=flags.rstrip(")"), font=("Segoe UI", 8), bg=t["line"], fg=t["muted"],
                     padx=self.px(5)).pack(side="left", padx=self.px(6))
        tk.Label(top, text=raw_at(self.raw_row, where, at), font=("Segoe UI", 9), bg=t["card"],
                 fg=t["faint"]).pack(side="right")
        row = tk.Frame(card, bg=t["card"])
        row.pack(fill="x", padx=self.px(8), pady=(self.px(4), self.px(7)))
        inner_w = self.panel_w - self.px(80)
        widths = {"name": 0.28, "star": 0.13, "items": 0.40, "there": 0.19}
        for c, (key, value) in enumerate(cells):
            part = key.split(".")[-1]
            row.grid_columnconfigure(c, weight=int(widths[part] * 100))
            chip, note = self._chip(row, key, value, t["card"], int(inner_w * widths[part]) - self.px(16))
            chip.grid(row=0, column=c, sticky="new", padx=(0, self.px(4)))
            self._note_slot[key] = lambda n=note, c=c: n.grid(row=1, column=c, sticky="new",
                                                              padx=(0, self.px(4)), pady=(self.px(3), 0))
        for w in [card, top, row, *top.winfo_children(), *row.winfo_children()]:
            w.bind("<Enter>", lambda e, h=(where, at): self.set_hover(h), add="+")
            w.bind("<Leave>", lambda e, c=card: self.root.after(40, self.clear_hover, c), add="+")

    def set_hover(self, h):
        if self.hover != h:
            self.hover = h
            if self.overlay:
                self.draw_frame()

    def clear_hover(self, card):
        w = self.root.winfo_containing(*self.root.winfo_pointerxy())
        if w is not None and str(w).startswith(str(card)):
            return
        if self.hover is not None:
            self.hover = None
            if self.overlay:
                self.draw_frame()

    def draw_fields(self, old: dict):
        for w in self.inner.winfo_children():
            w.destroy()
        self.chips, self.notes, self.missed, self.values, self._note_slot = {}, {}, {}, {}, {}
        by_group = defaultdict(list)
        units: dict[tuple, list] = {}
        labels = {}
        for key, group, label, value in self.fields:
            head = key.split(".")[0]
            if head in ("board", "bench"):
                where, at = key.split(".")[:2]
                units.setdefault((where, at), []).append((key, value))
                labels[(where, at)] = label
            else:
                by_group[head].append((key, label, value))

        stats = [c for h in ("stage", "gold", "hp", "level", "xp", "streak") for c in by_group[h]]
        self._grid(self._section("Round"), stats, 3)
        self._grid(self._section("Phase"), by_group["planning"] + by_group["home_board"], 2, captions=False)
        if by_group["shop"]:
            self._grid(self._section("Shop"), [(k, None, v) for k, _, v in by_group["shop"]], 3, captions=False)
        if by_group["item"]:
            self._grid(self._section("Item column", len(by_group["item"])),
                       [(k, None, v) for k, _, v in by_group["item"]], 3, captions=False)
        if by_group["augment"]:
            self._grid(self._section("Augments"), [(k, None, v) for k, _, v in by_group["augment"]], 1, captions=False)
        if by_group["lobby_hp"]:
            self._grid(self._section("Lobby HP"), by_group["lobby_hp"], 1, captions=False)
        for where in ("board", "bench"):
            mine = [(at, cells) for (w, at), cells in units.items() if w == where]
            body = self._section(where.capitalize(), len(mine))
            if not mine:
                tk.Label(body, text="nothing", font=("Segoe UI", 9), bg=self.t["panel"],
                         fg=self.t["faint"]).pack(anchor="w")
            for at, cells in mine:
                self._unit_card(body, where, at, labels[(where, at)], cells)

        body = self._section("Missed (units or items the state left out)")
        body.grid_columnconfigure(1, weight=1)
        for i, what in enumerate(MISSED):
            tk.Label(body, text=what, font=("Segoe UI", 9), bg=self.t["panel"], fg=self.t["muted"],
                     anchor="w").grid(row=i, column=0, sticky="w", padx=(0, self.px(8)), pady=self.px(2))
            e = tk.Entry(body, font=("Segoe UI", 10), bg=self.t["entry"], fg=self.t["text"],
                         insertbackground=self.t["text"], relief="flat", highlightthickness=1,
                         highlightbackground=self.t["line"], highlightcolor=self.t["muted"])
            e.insert(0, (old.get("missed") or {}).get(what, ""))
            e.grid(row=i, column=1, sticky="ew", pady=self.px(2))
            self.missed[what] = e
        tk.Frame(self.inner, bg=self.t["panel"], height=self.px(16)).pack()

        for key, note in self.notes.items():
            if key in old.get("wrong", {}):
                note.insert(0, old["wrong"][key].get("note", ""))
            if self.states[key] == WRONG and not key.endswith(".there"):
                self._note_slot[key]()
        self.list_canvas.yview_moveto(0)

    def cycle(self, key):
        nxt = {OK: WRONG, WRONG: UNSURE, UNSURE: OK}
        there = key.endswith(".there")
        if there:
            nxt = {OK: WRONG, WRONG: OK, UNSURE: OK}         # "not there": marked or not
        self.states[key] = nxt[self.states[key]]
        bg, fg = self.t["off"] if there and self.states[key] == OK else self.t[self.states[key]]
        self.chips[key].config(bg=bg, fg=fg, text=self._chip_text(key, self.values[key]))
        note = self.notes[key]
        if self.states[key] == WRONG and not there:
            self._note_slot[key]()
            note.focus_set()
        else:
            if note.winfo_manager() == "pack":
                note.pack_forget()
            elif note.winfo_manager() == "grid":
                note.grid_remove()
            self.root.focus_set()
        if key.startswith(("board.", "bench.")) and self.overlay:
            self.draw_frame()

    def collect(self) -> dict:
        shown = [k for k, _, _, _ in self.fields]
        value = {k: v for k, _, _, v in self.fields}
        return {
            "shown": shown,
            "wrong": {k: {"value": value[k], "note": self.notes[k].get().strip()}
                      for k in shown if self.states[k] == WRONG},
            "unsure": [k for k in shown if self.states[k] == UNSURE],
            "missed": {w: e.get().strip() for w, e in self.missed.items() if e.get().strip()},
            "checked": time.strftime("%Y-%m-%d %H:%M"),
        }

    def save(self):
        if self.pos < len(self.queue):
            self.changes[self.key()] = self.collect()
        merge_save(CHECKS, self.changes)

    def next(self):
        if self.pos >= len(self.queue):
            return
        self.save()
        self.pos += 1
        self.show()

    def back(self):
        if self.pos > 0:
            self.save()
            self.pos -= 1
            self.show()

    def toggle_overlay(self, event):
        if isinstance(event.widget, tk.Entry):
            return
        self.overlay = not self.overlay
        self.draw_frame()

    def toggle_theme(self, event):
        """Rebuild the window in the other theme; marks on the frame on screen are kept (not saved)."""
        if isinstance(event.widget, tk.Entry):
            return
        keep = self.collect() if self.pos < len(self.queue) else None
        self.theme = "light" if self.theme == "dark" else "dark"
        for w in self.root.winfo_children():
            w.destroy()
        self.build()
        self.show(keep)

    def draw_score(self):
        data = load(CHECKS)
        data.update(self.changes)
        lines = [f"{g:14s} {n - w:5d}/{n:<5d} {100 * (n - w) / n:5.1f}%" for g, n, w in report(data) if n]
        self.score.config(text=f"{len(data)} frames checked\n" + "\n".join(lines[:9]))

    def quit(self):
        """Saves what was confirmed; the frame on screen counts only once Enter is pressed on it."""
        merge_save(CHECKS, self.changes)
        self.root.destroy()


def load_states(vods: list[str] | None) -> tuple[dict, dict]:
    """(vod, game) -> {frame: cleaned row}, and the same for the raw rows, for games with full frames."""
    clean, raw = {}, {}
    for path in sorted(paths.STATES.glob("*/game_??.clean.jsonl")):
        vod, game = path.parent.name, path.name.split(".")[0]
        if vods and vod not in vods or not (paths.FRAMES / vod / game).is_dir():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        clean[(vod, game)] = {r["frame"]: r for r in rows if r.get("in_game", True)}
        raw_path = path.with_name(f"{game}.jsonl")
        if raw_path.exists():
            raw[(vod, game)] = {r["frame"]: r for r in
                                (json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l)}
    return clean, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vods", nargs="*", help="Only these VODs (default: every VOD with full frames)")
    parser.add_argument("--n", type=int, default=200, help="Frames in this session's queue")
    parser.add_argument("--planning", action="store_true", help="Planning frames only")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", action="store_true", help="Print accuracy per field and stop")
    theme = parser.add_mutually_exclusive_group()
    theme.add_argument("--dark", action="store_true", help="Dark theme (default: follow Windows)")
    theme.add_argument("--light", action="store_true", help="Light theme")
    args = parser.parse_args()

    checks = load(CHECKS)
    if args.report:
        print(f"{len(checks)} frames checked")
        for group, n, w in report(checks):
            print(f"  {group:14s} {n - w:6d}/{n:<6d} {100 * (n - w) / max(n, 1):5.1f}% right")
        missed = Counter(w for e in checks.values() for w in (e.get("missed") or {}))
        if missed:
            print("  frames with something missed:", dict(missed))
        return
    clean, raw = load_states(args.vods)
    if not clean:
        sys.exit("No cleaned games with full frames on disk (~/tft_frames_temp).")
    queue = sample({k: list(v.values()) for k, v in clean.items()}, args.n, set(checks), args.planning,
                   seed=args.seed if args.seed is not None else int(time.time()))
    print(f"{len(clean)} games, {len(queue)} frames in the queue")
    _dpi_aware()
    root = tk.Tk()
    Checker(root, queue, clean, raw, dark=args.dark or (not args.light and windows_dark()))
    root.mainloop()


if __name__ == "__main__":
    main()
