"""
Label unit crops with the champion they show.

Usage:
    python scripts/label_units.py
    python scripts/label_units.py --review        # go through already-labelled crops again
    python scripts/label_units.py --unsure        # only crops the model is unsure about, least sure first
    python scripts/label_units.py --unsure 0.7    # ...below 70% confidence (default 90%)

--unsure gives the most value per label: crops the model already gets
right with high confidence teach it little. Needs a trained model.

How it works:
  - The crop is shown large on the left; the frame it came from is shown on
    the right with the unit boxed, for context.
  - Type any part of the champion's name. Typos are fine: the list under the
    box shows the closest names. Click one, or press Enter for the first,
    or press its number (1-8).
  - "Looks like" buttons are the model's best guesses with its confidence
    (assets/unit_classifier.pt, trained by scripts/train_unit_classifier.py).
    With the box empty, Enter accepts the first one. Retrain now and then as
    labels pile up; without a trained model, suggestions fall back to the most
    similar crops labelled so far.

Keys:
    Enter        first search result (or first suggestion when the box is empty)
    1-8          that search result
    F1-F4        that suggestion
    Tab          not a unit / empty hex
    Esc          skip for now (comes back at the end)
    Ctrl+Z       undo the last label
    Ctrl+Q       save and quit   (every label is saved immediately anyway)

Labels are stored in data/processed/unit_crops/labels.json as
{crop file name: champion name}; "_none" marks a crop with no unit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import base64
import difflib
import json
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

from src.state.game_data import load_game_data
from src.vision.unit_classifier import UnitClassifier

CROP_DIR = Path("data/processed/unit_crops")
SEED_DIR = Path("data/processed/champion_crops")   # auto-labelled bench crops, used for suggestions
NONE = "_none"
MAX_RESULTS = 8
MAX_SUGGESTIONS = 4

BG, FG, DIM, ACCENT = "#1e1f24", "#e8e8ea", "#8b8d98", "#f2c14e"


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


def fingerprint(img: np.ndarray) -> np.ndarray:
    """Small normalised picture of the unit (health bar and floor edges trimmed off)."""
    h, w = img.shape[:2]
    body = img[int(h * 0.22):int(h * 0.95), int(w * 0.12):int(w * 0.88)]
    vec = cv2.resize(body, (20, 26), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def to_photo(img: np.ndarray) -> tk.PhotoImage:
    ok, buf = cv2.imencode(".png", img)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class Labeler:
    def __init__(self, review: bool, unsure: float | None = None):
        self.game = load_game_data()
        # Shop champions plus trait/augment units (Stonebark Tree, Training Dummy, ...)
        self.names = self.game.unit_names
        self.index = json.loads((CROP_DIR / "index.json").read_text(encoding="utf-8")) \
            if (CROP_DIR / "index.json").exists() else {}
        self.labels_path = CROP_DIR / "labels.json"
        self.labels: dict[str, str] = json.loads(self.labels_path.read_text(encoding="utf-8")) \
            if self.labels_path.exists() else {}

        files = sorted(p.name for p in (CROP_DIR / "images").glob("*.jpg"))
        self.queue = [f for f in files if (f in self.labels) == review]
        self.total = len(files)
        self.history: list[str] = []
        self.pos = 0

        # Known examples for "looks like" suggestions: labelled crops + auto-labelled bench crops.
        self.known_vecs: list[np.ndarray] = []
        self.known_names: list[str] = []
        for f, name in self.labels.items():
            if name != NONE:
                self._learn(CROP_DIR / "images" / f, name)
        if SEED_DIR.exists():
            for folder in SEED_DIR.iterdir():
                if folder.is_dir() and folder.name in self.names:
                    for p in folder.glob("*.jpg"):
                        self._learn(p, folder.name)

        self.classifier = UnitClassifier() if UnitClassifier.available() else None
        self.confidence: dict[str, float] = {}
        if unsure is not None:
            self.queue = self._unsure_first(self.queue, unsure)

        self._build_window()
        self.show()

    # ── data ──────────────────────────────────────────────────────────────────

    def _learn(self, path: Path, name: str):
        img = cv2.imread(str(path))
        if img is not None:
            self.known_vecs.append(fingerprint(img))
            self.known_names.append(name)

    def _unsure_first(self, files: list[str], below: float) -> list[str]:
        """The crops whose best guess is under `below`, least confident first."""
        if self.classifier is None:
            sys.exit("--unsure needs a trained model: python scripts/train_unit_classifier.py")
        print(f"Scoring {len(files)} crops with the classifier ...")
        imgs, kept = [], []
        for f in files:
            img = cv2.imread(str(CROP_DIR / "images" / f))
            if img is not None:
                imgs.append(img)
                kept.append(f)
        best = self.classifier.probabilities(imgs).max(axis=1) if imgs else np.zeros(0)
        order = [i for i in np.argsort(best) if best[i] < below]
        print(f"{len(order)} of {len(kept)} are below {below:.0%} confidence")
        return [kept[i] for i in order]

    def suggestions(self, img: np.ndarray) -> list[str]:
        self.confidence = {}
        if self.classifier is not None:
            preds = self.classifier.predict(img, top_k=MAX_SUGGESTIONS)
            self.confidence = dict(preds)
            return [name for name, prob in preds if prob >= 0.02]
        if not self.known_vecs:
            return []
        scores = np.stack(self.known_vecs) @ fingerprint(img)
        out: list[str] = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] < 0.35:
                break
            if self.known_names[i] not in out:
                out.append(self.known_names[i])
            if len(out) == MAX_SUGGESTIONS:
                break
        return out

    def save(self):
        self.labels_path.write_text(json.dumps(self.labels, indent=0, sort_keys=True), encoding="utf-8")

    # ── window ────────────────────────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Label units")
        self.root.configure(bg=BG)
        big = tkfont.Font(size=13)
        small = tkfont.Font(size=10)

        self.status = tk.Label(self.root, bg=BG, fg=DIM, font=small, anchor="w")
        self.status.grid(row=0, column=0, columnspan=2, sticky="we", padx=12, pady=(10, 4))

        self.crop_view = tk.Label(self.root, bg=BG)
        self.crop_view.grid(row=1, column=0, rowspan=4, padx=12, pady=6)
        self.context_view = tk.Label(self.root, bg=BG)
        self.context_view.grid(row=1, column=1, padx=12, pady=6, sticky="n")

        tk.Label(self.root, text="Looks like:", bg=BG, fg=DIM, font=small, anchor="w") \
            .grid(row=2, column=1, sticky="we", padx=12)
        self.suggest_row = tk.Frame(self.root, bg=BG)
        self.suggest_row.grid(row=3, column=1, sticky="we", padx=12)

        panel = tk.Frame(self.root, bg=BG)
        panel.grid(row=4, column=1, sticky="nwe", padx=12, pady=8)
        self.query = tk.StringVar()
        self.entry = tk.Entry(panel, textvariable=self.query, font=big, width=30,
                              bg="#2b2d35", fg=FG, insertbackground=FG, relief="flat")
        self.entry.pack(fill="x", ipady=6)
        self.query.trace_add("write", lambda *_: self.refresh_results())
        self.result_box = tk.Frame(panel, bg=BG)
        self.result_box.pack(fill="x", pady=(6, 0))
        self.big, self.small = big, small

        actions = tk.Frame(self.root, bg=BG)
        actions.grid(row=5, column=0, columnspan=2, sticky="we", padx=12, pady=(4, 12))
        for text, command in (("Not a unit  (Tab)", self.mark_none), ("Skip  (Esc)", self.skip),
                              ("Undo  (Ctrl+Z)", self.undo), ("Quit  (Ctrl+Q)", self.quit)):
            tk.Button(actions, text=text, command=command, font=small, bg="#33353f", fg=FG,
                      activebackground="#444754", activeforeground=FG, relief="flat", padx=10, pady=4) \
                .pack(side="left", padx=(0, 8))

        self.root.bind("<Return>", lambda e: self.pick_first())
        self.root.bind("<Tab>", lambda e: (self.mark_none(), "break")[1])
        self.root.bind("<Escape>", lambda e: self.skip())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-q>", lambda e: self.quit())
        for n in range(1, MAX_RESULTS + 1):
            self.root.bind(str(n), lambda e, k=n: (self.pick_result(k - 1), "break")[1])
        for n in range(1, MAX_SUGGESTIONS + 1):
            self.root.bind(f"<F{n}>", lambda e, k=n: self.pick_suggestion(k - 1))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _buttons(self, parent: tk.Frame, names: list[str], prefix: str, command):
        for child in parent.winfo_children():
            child.destroy()
        for i, name in enumerate(names):
            cost = self.game.cost(name)
            note = f"({cost}g)" if cost else "(trait unit)"
            text = "not a unit" if name == NONE else f"{name}  {note}"
            if parent is self.suggest_row and name in self.confidence:
                text += f"  {self.confidence[name]:.0%}"
            tk.Button(parent, text=f"{prefix}{i + 1}  {text}", anchor="w",
                      command=lambda n=name: command(n), font=self.big, bg="#2b2d35",
                      fg=ACCENT if i == 0 else FG, activebackground="#444754", activeforeground=FG,
                      relief="flat", padx=10, pady=3) \
                .pack(fill="x" if parent is self.result_box else None,
                      side="top" if parent is self.result_box else "left", padx=(0, 6), pady=2)

    # ── flow ──────────────────────────────────────────────────────────────────

    def show(self):
        if self.pos >= len(self.queue):
            self.status.config(text=f"All done - {len(self.labels)} of {self.total} crops labelled. "
                                    "Close the window.")
            self.crop_view.config(image="")
            self.context_view.config(image="")
            self._buttons(self.suggest_row, [], "", self.assign)
            self._buttons(self.result_box, [], "", self.assign)
            return
        name = self.queue[self.pos]
        img = cv2.imread(str(CROP_DIR / "images" / name))
        info = self.index.get(name, {})

        self.crop_photo = to_photo(cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        self.crop_view.config(image=self.crop_photo)

        frame = cv2.imread(info["frame_path"]) if info.get("frame_path") else None
        if frame is not None:
            x, y, w, h = info["window"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
            self.context_photo = to_photo(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
            self.context_view.config(image=self.context_photo)
        else:
            self.context_view.config(image="")

        current = self.labels.get(name)
        where = info.get("where", "")
        stage = f"stage {int(info['stage'])}-{int(info['round'])}" \
            if info.get("stage") is not None and info.get("round") is not None else ""
        self.status.config(text=f"{len(self.labels)} of {self.total} labelled   |   "
                                f"{len(self.queue) - self.pos} left in this session   |   "
                                f"{where} {stage}" + (f"   |   current label: {current}" if current else ""))

        self.current_suggestions = self.suggestions(img)
        self._buttons(self.suggest_row, self.current_suggestions, "F", self.assign)
        self.query.set("")
        self.refresh_results()
        self.entry.focus_set()

    def refresh_results(self):
        self.results = search_champions(self.query.get(), self.names)
        self._buttons(self.result_box, self.results, "", self.assign)

    def assign(self, label: str):
        if self.pos >= len(self.queue):
            return
        name = self.queue[self.pos]
        self.labels[name] = label
        self.history.append(name)
        self.save()
        if label != NONE:
            self._learn(CROP_DIR / "images" / name, label)
        self.pos += 1
        self.show()

    def pick_first(self):
        if self.results:
            self.assign(self.results[0])
        elif not self.query.get().strip() and self.current_suggestions:
            self.assign(self.current_suggestions[0])

    def pick_result(self, i: int):
        if i < len(self.results):
            self.assign(self.results[i])

    def pick_suggestion(self, i: int):
        if i < len(self.current_suggestions):
            self.assign(self.current_suggestions[i])

    def mark_none(self):
        self.assign(NONE)

    def skip(self):
        if self.pos < len(self.queue):
            self.queue.append(self.queue.pop(self.pos))
            self.show()

    def undo(self):
        if not self.history:
            return
        name = self.history.pop()
        self.labels.pop(name, None)
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
    parser.add_argument("--review", action="store_true", help="Show already-labelled crops instead")
    parser.add_argument("--unsure", type=float, nargs="?", const=0.9, default=None, metavar="BELOW",
                        help="Only crops whose best guess is below this confidence (default 0.9), "
                             "least sure first")
    args = parser.parse_args()

    if not (CROP_DIR / "images").exists() or not any((CROP_DIR / "images").glob("*.jpg")):
        sys.exit(f"No crops in {CROP_DIR / 'images'} - run scripts/collect_unit_crops.py first")
    app = Labeler(args.review, args.unsure)
    app.root.mainloop()
    print(f"{len(app.labels)} of {app.total} crops labelled -> {app.labels_path}")


if __name__ == "__main__":
    main()
