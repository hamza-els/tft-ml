"""
Rebuild the item-column templates (assets/templates/items_seen/) from scratch with the
current ItemColumnReader, keeping every name already given, then re-read the column.

Why: before 2026-11-22 the reader cut slots at a fixed place, so dragged / hovered items
and the column's occasional 4 px shift gave half-cut templates. The reader now finds the
column per frame and cuts only settled slots; this redoes the templates with it.

Steps (all by default, or pick with --steps):
  harvest  cut a fresh template set from every VOD with full frames (harvest_item_icons.py,
           into a new folder; named from official art when it matches, else item_XXX)
  name     carry names over from the old set, by template likeness (>= seen_threshold):
           Hamza's answers in names_review.json first (a new template like one he marked
           "not an item" is dropped), then names already on old templates
  swap     old set (and names_review.json) -> ~/tft_data/backup/items_seen_<time>/, new set in
  reread   item column of every VOD with full frames read again (harvest_item_icons.py
           --update-states; no new templates expected, same frames)
  remap    VODs without full frames: their stored readings name old templates; each old
           template is matched to the new set (unmatched -> None, an unreadable item)
  finish   clean_states, build_dataset, upload_states for every VOD touched

Usage:
    python scripts/rebuild_item_templates.py
    python scripts/rebuild_item_templates.py --steps reread,remap,finish --work DIR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import shutil
import subprocess
import time

import cv2

from src.state.game_data import load_game_data
from src.utils import paths
from src.utils.config import vision_cfg
from src.vision.items import ItemColumnReader, _TemplateSet

ROOT = Path(__file__).resolve().parents[1]
SEEN = ROOT / vision_cfg["item_column"]["seen_dir"]
ANSWERS_NAME = "names_review.json"
NONE, UNKNOWN = "_none", "_unknown"
STEPS = ["harvest", "name", "swap", "reread", "remap", "finish"]


def base(name: str) -> str:
    return name.split("~")[0]


def run(cmd: list[str]) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print("  $", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def vods() -> tuple[list[str], list[str]]:
    """(VODs with state files and full frames, VODs with state files only)."""
    full, slim = [], []
    for d in sorted(p for p in paths.STATES.iterdir() if p.is_dir()):
        if not list(d.glob("game_??.jsonl")):
            continue
        (full if (paths.FRAMES / d.name).is_dir() and list((paths.FRAMES / d.name).glob("game_??"))
         else slim).append(d.name)
    return full, slim


def template_set(folder: Path) -> _TemplateSet:
    r = ItemColumnReader()
    s = _TemplateSet(r.icon_size, int(vision_cfg["item_column"]["match_margin"]))
    s.load(folder)
    return s


def old_labels(folder: Path) -> tuple[dict[str, str], set[str]]:
    """Old template -> its name: Hamza's answer, else the file's own name; unnamed and
    'can't tell' left out. Bundle names folded."""
    aliases = load_game_data().item_aliases
    try:
        answers = json.loads((folder / ANSWERS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        answers = {}
    out = {}
    for p in folder.glob("*.png"):
        label = answers.get(p.stem) or (None if p.stem.startswith("item_") else base(p.stem))
        if not label or label == UNKNOWN:
            continue
        if label in aliases:
            label = aliases[label] or NONE
        out[p.stem] = label
    return out, set(answers)


def step_harvest(work: Path, full: list[str]) -> None:
    new = work / "new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    for v in full:
        print(f"harvest {v}", flush=True)
        run([sys.executable, "scripts/harvest_item_icons.py", "--frames", str(paths.FRAMES / v),
             "--seen-dir", str(new)])
    print(f"{len(list(new.glob('*.png')))} templates cut")


def step_name(work: Path) -> None:
    new = work / "new"
    labels, answered = old_labels(SEEN)
    old = template_set(SEEN)
    thr = float(vision_cfg["item_column"]["seen_threshold"])
    rows = {n: r for n, r in zip(old.names, old._rows)}
    hamza = [n for n in old.names if n in answered and n in labels]
    other = [n for n in old.names if n in labels and n not in answered]
    removed = work / "new_removed"
    removed.mkdir(exist_ok=True)
    taken = {p.stem for p in new.glob("*.png")}
    stats = {"hamza": 0, "old name": 0, "dropped (not an item)": 0, "kept": 0}
    log = {}
    for p in sorted(new.glob("*.png")):
        vec = old.prepare(cv2.imread(str(p)))
        label = None
        for group, key in ((hamza, "hamza"), (other, "old name")):
            if not group:
                continue
            scores = [(float(rows[n] @ vec), n) for n in group]
            s, n = max(scores)
            if s >= thr:
                label, how = labels[n], key
                break
        if label is None or (label == base(p.stem) and label != NONE):
            stats["kept"] += 1
            continue
        taken.discard(p.stem)
        if label == NONE:
            shutil.move(str(p), removed / p.name)
            stats["dropped (not an item)"] += 1
            log[p.stem] = NONE
            continue
        name, k = label, 2
        while name in taken:
            name, k = f"{label}~{k}", k + 1
        p.rename(new / f"{name}.png")
        taken.add(name)
        stats[how] += 1
        log[p.stem] = name
    (work / "name_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    left = sum(1 for p in new.glob("item_*.png"))
    print(f"named: {stats}; {left} still unnamed (item_XXX)")


def step_swap(work: Path) -> None:
    backup = paths.LOCAL / "backup" / f"items_seen_{time.strftime('%Y-%m-%d_%H%M')}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SEEN, backup)
    (work / "backup_path.txt").write_text(str(backup), encoding="utf-8")
    for p in SEEN.iterdir():
        if p.is_file():
            p.unlink()
    for p in (work / "new").glob("*.png"):
        shutil.copy2(p, SEEN / p.name)
    print(f"old set -> {backup}; {len(list(SEEN.glob('*.png')))} templates in {SEEN}")


def step_reread(full: list[str]) -> None:
    for v in full:
        print(f"reread {v}", flush=True)
        run([sys.executable, "scripts/harvest_item_icons.py", "--frames", str(paths.FRAMES / v),
             "--update-states", str(paths.STATES / v)])


def step_remap(work: Path, slim: list[str]) -> None:
    backup = Path((work / "backup_path.txt").read_text(encoding="utf-8"))
    old, new = template_set(backup), template_set(SEEN)
    aliases = load_game_data().item_aliases
    thr = float(vision_cfg["item_column"]["seen_threshold"])
    # Names given to old templates (Hamza's answers in the backed-up names_review.json, then
    # file names) win over likeness to the new set: an answered template was often a shifted
    # or overlaid cut that matches no clean new template, but its readings are still that item.
    labels, _ = old_labels(backup)
    mapping: dict[str, str | None] = {}
    for n, row in zip(old.names, old._rows):
        b = base(n)
        if n in labels:
            mapping[n if n.startswith("item_") else b] = None if labels[n] == NONE else labels[n]
            continue
        if not n.startswith("item_"):
            mapping[b] = aliases[b] if b in aliases else b      # already a name: keep it
            continue
        name, s = max(((nn, float(r @ row)) for nn, r in zip(new.names, new._rows)), key=lambda t: t[1])
        mapping[n] = base(name) if s >= thr else None
    (work / "remap.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    print(f"old item_XXX: {sum(1 for k, v in mapping.items() if k.startswith('item_') and v)} matched, "
          f"{sum(1 for k, v in mapping.items() if k.startswith('item_') and not v)} -> None")
    for v in slim:
        changed = 0
        for f in sorted((paths.STATES / v).glob("game_??.jsonl")):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l]
            for r in rows:
                bi = r.get("bench_items")
                if not bi:
                    continue
                nb = [mapping.get(n, n) if n else None for n in bi]
                if nb != bi:
                    r["bench_items"] = nb
                    changed += 1
            f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"  {v}: {changed} rows changed")


def step_finish(vod_list: list[str]) -> None:
    for v in vod_list:
        states = str(paths.STATES / v)
        run([sys.executable, "scripts/clean_states.py", "--states", states])
        run([sys.executable, "scripts/build_dataset.py", "--states", states])
        run([sys.executable, "scripts/upload_states.py", "--", v])  # ids may start with "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--work", default=str(paths.LOCAL / "item_template_rebuild"))
    args = parser.parse_args()
    steps = args.steps.split(",")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    full, slim = vods()
    print(f"{len(full)} VODs with full frames, {len(slim)} without")
    t0 = time.time()
    for step in STEPS:
        if step not in steps:
            continue
        print(f"\n=== {step} ({(time.time() - t0) / 60:.0f} min) ===", flush=True)
        if step == "harvest":
            step_harvest(work, full)
        elif step == "name":
            step_name(work)
        elif step == "swap":
            step_swap(work)
        elif step == "reread":
            step_reread(full)
        elif step == "remap":
            step_remap(work, slim)
        elif step == "finish":
            step_finish(full + slim)
    print(f"\nDone ({(time.time() - t0) / 60:.0f} min).")


if __name__ == "__main__":
    main()
