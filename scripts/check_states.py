"""
Check cleaned states against the full frame, by eye (Hamza's checker, step 5b).

Usage:
    python scripts/check_states.py                    # random in-game frames of the VODs with full frames
    python scripts/check_states.py --planning         # planning frames only (where units are read)
    python scripts/check_states.py --vods knB6dE8Rbgg --n 100
    python scripts/check_states.py --report           # accuracy per field from the checks so far

The frame is shown on the left (cleaned board and bench units drawn on their
hexes and slots; O toggles that; a 3x zoom follows the mouse), every cleaned
field on the right as chips. Everything unmarked counts as right when you go
on. Click a chip: wrong (red, a box opens for what it should be, optional);
again: can't tell (yellow, left out of the figures); again: back to right.
Units have one chip each for champion, star and items, and "not there" for a
unit that isn't on that spot; "missed" boxes take units or items the state
left out. Next to each unit, in grey, what that single frame read at that
spot (so a mistake can be put on the reading or on the cleaning).

Keys: Enter = next (saves this frame), Ctrl+Backspace = back, O = overlay, Esc = quit (the
frame on screen is not saved unless Enter was pressed on it).

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
SCALE = 2 / 3                       # 1920x1080 -> 1280x720
ZOOM, ZOOM_W, ZOOM_H = 3, 160, 100  # 3x of a 160x100 patch under the mouse
OK, WRONG, UNSURE = "ok", "wrong", "unsure"
COLOURS = {OK: "#d9ead3", WRONG: "#f4a6a6", UNSURE: "#ffe599"}
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

class Checker:
    def __init__(self, root: tk.Tk, queue: list[tuple], clean: dict, raw: dict):
        from src.vision.bench import BenchSlots
        from src.vision.board import BoardHexes
        self.board, self.bench = BoardHexes(), BenchSlots()
        self.root, self.queue, self.clean, self.raw = root, queue, clean, raw
        self.pos = 0
        self.changes: dict = {}
        self.overlay = True
        self.states: dict[str, str] = {}
        self.notes: dict[str, tk.Entry] = {}
        self.missed: dict[str, tk.Entry] = {}
        self.img = None

        root.title("State checker")
        left = tk.Frame(root)
        left.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Label(left)
        self.canvas.pack()
        self.canvas.bind("<Motion>", self.zoom_at)
        self.status = tk.Label(left, font=("Segoe UI", 11), anchor="w", justify="left")
        self.status.pack(fill="x", pady=(6, 0))
        self.score = tk.Label(left, font=("Consolas", 9), anchor="w", justify="left", fg="#444")
        self.score.pack(fill="x")
        tk.Label(left, text="Enter = next (unmarked = right)   click a chip: wrong / can't tell / right   "
                            "O = overlay   Ctrl+Backspace = back   Esc = quit (frame on screen not saved)",
                 fg="#666").pack(anchor="w")

        right = tk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew", padx=8)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self.zoom = tk.Label(right, bg="black")
        self.zoom.pack(anchor="w")
        box = tk.Frame(right)
        box.pack(fill="both", expand=True, pady=(6, 0))
        self.list_canvas = tk.Canvas(box, width=600, highlightthickness=0)
        bar = tk.Scrollbar(box, orient="vertical", command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.list_canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.list_canvas)
        self.list_canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")))
        self.list_canvas.bind_all("<MouseWheel>", lambda e: self.list_canvas.yview_scroll(-e.delta // 120, "units"))

        root.bind("<Return>", lambda e: self.next())
        root.bind("<Control-BackSpace>", lambda e: None if isinstance(e.widget, tk.Entry) else self.back())
        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<Key-o>", self.toggle_overlay)
        root.bind("<Key-O>", self.toggle_overlay)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.show()

    # one frame
    def key(self):
        vod, game, frame = self.queue[self.pos]
        return f"{vod}/{game}/{frame}"

    def show(self):
        if self.pos >= len(self.queue):
            self.status.config(text="All done.")
            return
        vod, game, frame = self.queue[self.pos]
        self.row = self.clean[(vod, game)][frame]
        self.raw_row = self.raw.get((vod, game), {}).get(frame)
        self.frame = cv2.imread(str(paths.FRAMES / vod / game / f"frame_{frame:06d}.jpg"))
        if self.frame is None:
            self.frame = np.zeros((1080, 1920, 3), np.uint8)
        old = self.changes.get(self.key()) or load(CHECKS).get(self.key()) or {}
        self.fields = fields_of(self.row)
        self.states = {k: OK for k, _, _, _ in self.fields}
        self.states.update({k: WRONG for k in old.get("wrong", {})})
        self.states.update({k: UNSURE for k in old.get("unsure", [])})
        self.draw_frame()
        self.draw_fields(old)
        self.status.config(text=f"{self.pos + 1}/{len(self.queue)}   {vod} {game} frame {frame} "
                                f"({frame / 2 / 60:.1f} min)   augments from {self.row.get('augment_source', '?')}")
        self.draw_score()

    def draw_frame(self):
        img = self.frame.copy()
        if self.overlay:
            for u in self.row.get("board") or []:
                cx, cy = self.board.center(*u["hex"])
                self._tag(img, cx, cy + 25, u)
            for u in self.row.get("bench") or []:
                cx = int(self.bench.center_x(u["slot"]))
                self._tag(img, cx, self.bench.region(u["slot"])[1] + 120, u)
        small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        self.img = to_photo(small)
        self.canvas.config(image=self.img)

    @staticmethod
    def _tag(img, cx, y, u):
        text = f"{u.get('name', '?')} {u.get('star') or '?'}*" + (f" +{len(u['items'])}" if u.get("items") else "")
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = int(cx - w / 2)
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + 4), (0, 0, 0), -1)
        colour = (0, 200, 255) if u.get("carried") or u.get("filled") else (255, 255, 255)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    def zoom_at(self, event):
        x, y = int(event.x / SCALE), int(event.y / SCALE)
        x0 = min(max(0, x - ZOOM_W // 2), 1920 - ZOOM_W)
        y0 = min(max(0, y - ZOOM_H // 2), 1080 - ZOOM_H)
        patch = cv2.resize(self.frame[y0:y0 + ZOOM_H, x0:x0 + ZOOM_W], None, fx=ZOOM, fy=ZOOM,
                           interpolation=cv2.INTER_NEAREST)
        self.zoom_img = to_photo(patch)
        self.zoom.config(image=self.zoom_img)

    def draw_fields(self, old: dict):
        for w in self.inner.winfo_children():
            w.destroy()
        self.notes, self.missed = {}, {}
        rows: dict[str, list] = defaultdict(list)          # one line per label (a unit = one line)
        order = []
        for key, group, label, value in self.fields:
            if label not in rows:
                order.append(label)
            rows[label].append((key, value))
        for r, label in enumerate(order):
            line = tk.Frame(self.inner)
            line.grid(row=r, column=0, sticky="w", pady=1)
            tk.Label(line, text=label, width=22, anchor="w").pack(side="left")
            for key, value in rows[label]:
                b = tk.Button(line, text=value, relief="flat", padx=6, wraplength=420,
                              bg=COLOURS[self.states[key]] if not key.endswith(".there")
                              else (COLOURS[WRONG] if self.states[key] == WRONG else "#eeeeee"))
                b.config(command=lambda k=key, btn=b: self.cycle(k, btn))
                b.pack(side="left", padx=2)
            first = rows[label][0][0]
            if first.startswith(("board.", "bench.")):
                where, at = first.split(".")[:2]
                tk.Label(line, text=raw_at(self.raw_row, where, at), fg="#888").pack(side="left", padx=6)
            for key, _ in rows[label]:
                note = tk.Entry(line, width=18)
                if key in old.get("wrong", {}):
                    note.insert(0, old["wrong"][key].get("note", ""))
                self.notes[key] = note
                if self.states[key] == WRONG:
                    note.pack(side="left", padx=2)
        base = len(order)
        for i, what in enumerate(MISSED):
            line = tk.Frame(self.inner)
            line.grid(row=base + i, column=0, sticky="w", pady=(8 if i == 0 else 1, 1))
            tk.Label(line, text=f"missed on {what}", width=22, anchor="w").pack(side="left")
            e = tk.Entry(line, width=50)
            e.insert(0, (old.get("missed") or {}).get(what, ""))
            e.pack(side="left")
            self.missed[what] = e
        self.list_canvas.yview_moveto(0)

    def cycle(self, key, button):
        nxt = {OK: WRONG, WRONG: UNSURE, UNSURE: OK}
        if key.endswith(".there"):
            nxt = {OK: WRONG, WRONG: OK, UNSURE: OK}         # "not there": marked or not
        self.states[key] = nxt[self.states[key]]
        colour = COLOURS[self.states[key]]
        if key.endswith(".there") and self.states[key] == OK:
            colour = "#eeeeee"
        button.config(bg=colour)
        note = self.notes[key]
        if self.states[key] == WRONG and not key.endswith(".there"):
            note.pack(side="left", padx=2, after=button)
            note.focus_set()
        else:
            note.pack_forget()
            self.root.focus_set()

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

    def draw_score(self):
        data = load(CHECKS)
        data.update(self.changes)
        lines = [f"{g:14s} {n - w:5d}/{n:<5d} {100 * (n - w) / n:5.1f}% right" for g, n, w in report(data) if n]
        self.score.config(text=f"{len(data)} frames checked\n" + "\n".join(lines[:8]))

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
    root = tk.Tk()
    root.geometry("1920x1000+0+0")
    Checker(root, queue, clean, raw)
    root.mainloop()


if __name__ == "__main__":
    main()
