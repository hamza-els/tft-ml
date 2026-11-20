"""
Label unit crops with the champion they show.

Usage:
    python scripts/label_units.py
    python scripts/label_units.py --review        # go through already-labelled crops again
    python scripts/label_units.py --unsure        # only crops the model is unsure about, least sure first
    python scripts/label_units.py --unsure 0.7    # ...below 70% confidence (default 90%)

    python scripts/label_units.py --relabel _none   # go through every crop labelled "not a unit" again
    python scripts/label_units.py --relabel Ashe    # ...or any other label

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
    Tab          not a unit / empty hex (scenery, an effect, the Little Legend)
    Shift+Tab    a unit, but it can't be told which (mid-animation, effects)
    Ctrl+T       covered by a tooltip (dark panel with text over the unit)
    Esc          skip for now (comes back at the end)
    Ctrl+Z       undo the last label
    Ctrl+Q       save and quit   (every label is saved immediately anyway)

Labels are stored in data/processed/unit_crops/labels.json as
{crop file name: champion name}; "_none" marks a crop with no unit,
"_unknown" a unit that can't be recognised, "_covered" a crop hidden by a
tooltip. The classifier learns all three: "_unknown" still counts as an
occupied hex in the game states, "_covered" as no information at all.
"_wrong_frame" (Ctrl+W) marks every crop of a frame that should not have been
used (combat, someone else's board, a round transition): it is not a class,
training skips it, and it is kept to check the planning / own-board detection.
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

from scripts.label_common import MAX_RESULTS, search_champions, to_photo
from src.utils import paths
from src.state.game_data import load_game_data
from tqdm import tqdm

from src.vision.unit_classifier import WEIGHTS, UnitClassifier

CROP_DIR = paths.CROP_DIR          # index.json, labels.json (repo)
IMAGES = paths.CROP_IMAGES         # crop images (local disk, src/utils/paths.py)
SCORE_CACHE = paths.CROP_SCORE_CACHE   # --unsure scores (local disk); rebuilt when the model changes
SEED_DIR = Path("data/processed/champion_crops")   # auto-labelled bench crops, used for suggestions
NONE = "_none"
UNKNOWN = "_unknown"
COVERED = "_covered"
WRONG_FRAME = "_wrong_frame"
SPECIAL = {NONE: "not a unit", UNKNOWN: "unit, can't tell which", COVERED: "covered by a tooltip",
           WRONG_FRAME: "wrong frame (combat / away / transition)"}
MAX_SUGGESTIONS = 4

BG, FG, DIM, ACCENT = "#1e1f24", "#e8e8ea", "#8b8d98", "#f2c14e"


def fingerprint(img: np.ndarray) -> np.ndarray:
    """Small normalised picture of the unit (health bar and floor edges trimmed off)."""
    h, w = img.shape[:2]
    body = img[int(h * 0.22):int(h * 0.95), int(w * 0.12):int(w * 0.88)]
    vec = cv2.resize(body, (20, 26), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Labeler:
    def __init__(self, review: bool, unsure: float | None = None, relabel: str | None = None):
        self.game = load_game_data()
        # Shop champions plus trait/augment units (Stonebark Tree, Training Dummy, ...)
        self.names = self.game.unit_names
        self.index = json.loads((CROP_DIR / "index.json").read_text(encoding="utf-8")) \
            if (CROP_DIR / "index.json").exists() else {}
        self.labels_path = CROP_DIR / "labels.json"
        self.labels: dict[str, str] = json.loads(self.labels_path.read_text(encoding="utf-8")) \
            if self.labels_path.exists() else {}

        files = sorted(p.name for p in (IMAGES).glob("*.jpg"))
        if relabel is not None:
            self.queue = [f for f in files if self.labels.get(f) == relabel]
            print(f"{len(self.queue)} crops labelled {relabel!r}")
        else:
            self.queue = [f for f in files if (f in self.labels) == review]
        self.total = len(files)
        self.history: list[str] = []
        self.changed: dict[str, str | None] = {}   # labels set (or removed by undo) this session
        self.pos = 0

        self.classifier = UnitClassifier() if UnitClassifier.available() else None
        self.confidence: dict[str, float] = {}

        # Without a model, "looks like" falls back to the most similar crops
        # labelled so far (labelled crops + auto-labelled bench crops). Reading
        # them all is slow on the Drive folder, so only done when needed.
        self.known_vecs: list[np.ndarray] = []
        self.known_names: list[str] = []
        if self.classifier is None:
            print("No trained model: loading labelled crops for suggestions ...")
            for f, name in self.labels.items():
                if name != NONE:
                    self._learn(IMAGES / f, name)
            if SEED_DIR.exists():
                for folder in SEED_DIR.iterdir():
                    if folder.is_dir() and folder.name in self.names:
                        for p in folder.glob("*.jpg"):
                            self._learn(p, folder.name)
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
        # Scores are kept in a cache tied to the model file, so only crops
        # added since the last run (or since retraining) are read again.
        model_id = f"{WEIGHTS.stat().st_mtime:.0f}"
        try:
            cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        if cache.get("model") != model_id:
            cache = {"model": model_id, "scores": {}}
        scores: dict[str, float] = cache["scores"]
        todo = [f for f in files if f not in scores]
        print(f"Scoring {len(todo)} crops with the classifier ({len(files) - len(todo)} already scored) ...")
        for start in tqdm(range(0, len(todo), 64), unit="batch", dynamic_ncols=True):
            imgs, names = [], []
            for f in todo[start:start + 64]:
                img = cv2.imread(str(IMAGES / f))
                if img is not None:
                    imgs.append(img)
                    names.append(f)
            if imgs:
                for f, p in zip(names, self.classifier.probabilities(imgs).max(axis=1)):
                    scores[f] = round(float(p), 4)
        try:
            SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SCORE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
        unsure = sorted((scores[f], f) for f in files if f in scores and scores[f] < below)
        print(f"{len(unsure)} of {len(files)} are below {below:.0%} confidence")
        return [f for _, f in unsure]

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
        """
        Write this session's changes into labels.json as it is on disk now:
        other scripts (prune_unit_crops.py) may have changed it meanwhile, and
        writing the whole in-memory copy would undo them.
        """
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
        for text, command in (("Not a unit  (Tab)", self.mark_none),
                              ("Unit, can't tell which  (Shift+Tab)", self.mark_unknown),
                              ("Covered by tooltip  (Ctrl+T)", self.mark_covered),
                              ("Wrong frame: combat / away  (Ctrl+W)", self.mark_wrong_frame), ("Skip  (Esc)", self.skip),
                              ("Undo  (Ctrl+Z)", self.undo), ("Quit  (Ctrl+Q)", self.quit)):
            tk.Button(actions, text=text, command=command, font=small, bg="#33353f", fg=FG,
                      activebackground="#444754", activeforeground=FG, relief="flat", padx=10, pady=4) \
                .pack(side="left", padx=(0, 8))

        self.root.bind("<Return>", lambda e: self.pick_first())
        self.root.bind("<Tab>", lambda e: (self.mark_none(), "break")[1])
        # Windows sends Shift+Tab as ISO_Left_Tab on some keyboards
        self.root.bind("<Shift-Tab>", lambda e: (self.mark_unknown(), "break")[1])
        try:
            self.root.bind("<ISO_Left_Tab>", lambda e: (self.mark_unknown(), "break")[1])
        except tk.TclError:
            pass
        self.root.bind("<Escape>", lambda e: self.skip())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-t>", lambda e: (self.mark_covered(), "break")[1])
        self.root.bind("<Control-w>", lambda e: (self.mark_wrong_frame(), "break")[1])
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
            text = SPECIAL[name] if name in SPECIAL else f"{name}  {note}"
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
        img = cv2.imread(str(IMAGES / name))
        while img is None:                  # moved away (prune_unit_crops.py) since the queue was built
            self.queue.pop(self.pos)
            if self.pos >= len(self.queue):
                self.show()
                return
            name = self.queue[self.pos]
            img = cv2.imread(str(IMAGES / name))
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
        self.changed[name] = label
        self.history.append(name)
        self.save()
        if label not in SPECIAL and self.classifier is None:
            self._learn(IMAGES / name, label)
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

    def mark_unknown(self):
        self.assign(UNKNOWN)

    def mark_covered(self):
        self.assign(COVERED)

    def mark_wrong_frame(self):
        """Label every crop of this crop's frame "_wrong_frame" and take them out of the queue."""
        if self.pos >= len(self.queue):
            return
        frame = self.index.get(self.queue[self.pos], {}).get("frame_path")
        group = [n for n, info in self.index.items() if frame and info.get("frame_path") == frame
                 and n not in self.labels] or [self.queue[self.pos]]
        for name in group:
            self.labels[name] = WRONG_FRAME
            self.changed[name] = WRONG_FRAME
        self.history.append(group)          # one undo puts the whole frame back
        self.save()
        later = set(group)
        self.queue = self.queue[:self.pos] + [n for n in self.queue[self.pos:] if n not in later]
        self.show()

    def skip(self):
        if self.pos < len(self.queue):
            self.queue.append(self.queue.pop(self.pos))
            self.show()

    def undo(self):
        if not self.history:
            return
        entry = self.history.pop()
        group = entry if isinstance(entry, list) else [entry]     # a "wrong frame" is one undo
        for name in group:
            self.labels.pop(name, None)
            self.changed[name] = None
        self.save()
        name = group[0]
        if name in self.queue:
            self.pos = self.queue.index(name)
        else:
            self.queue.insert(self.pos, name)
        for other in group[1:]:
            if other not in self.queue:
                self.queue.insert(self.pos + 1, other)
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
    parser.add_argument("--relabel", metavar="LABEL",
                        help="Go through every crop with this label again (e.g. _none)")
    args = parser.parse_args()

    if not (IMAGES).exists() or not any((IMAGES).glob("*.jpg")):
        sys.exit(f"No crops in {IMAGES} - run scripts/collect_unit_crops.py first")
    app = Labeler(args.review, args.unsure, args.relabel)
    app.root.mainloop()
    print(f"{len(app.labels)} of {app.total} crops labelled -> {app.labels_path}")


if __name__ == "__main__":
    main()
