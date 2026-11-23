"""
Apply the item template check (data/raw/review/item_check.html -> item_check.json).

    python scripts/apply_item_check.py ~/Downloads/item_check.json --dry-run
    python scripts/apply_item_check.py ~/Downloads/item_check.json

The JSON is {template: right name | "_none" (not an item) | "_unknown" (an item, can't tell)}.

1. Templates in assets/templates/items_seen/ renamed (Name, Name~2, ...); "_unknown" ones
   become item_XXX; "_none" ones move to items_seen_removed/.
2. VODs with full frames: the item column is read again with the corrected templates
   (harvest_item_icons.py --update-states --no-harvest).
3. VODs without full frames: stored readings hold item names, not templates, so a reading
   can be fixed only when every template of that name got the same new answer; other
   readings keep the old name (counted in the output).
4. clean_states, build_dataset, upload_states for every VOD touched.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import shutil
import subprocess

from src.state.game_data import load_game_data
from src.utils import paths
from src.utils.config import vision_cfg

ROOT = paths.REPO
SEEN = ROOT / vision_cfg["item_column"]["seen_dir"]
REMOVED = SEEN.parent / "items_seen_removed"
NONE, UNKNOWN = "_none", "_unknown"


def base(name: str) -> str:
    return name.split("~")[0]


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-finish", action="store_true", help="Skip re-reading and clean/encode/upload")
    args = parser.parse_args()

    game = load_game_data()
    valid = set(game.items) | set(game.emblems)
    answers = json.loads(Path(args.result).expanduser().read_text(encoding="utf-8"))
    stems = {p.stem for p in SEEN.glob("*.png")}
    bad = [t for t in answers if t not in stems] + [v for v in answers.values() if v not in valid | {NONE, UNKNOWN}]
    if bad:
        sys.exit(f"unknown templates or names: {bad}")
    answers = {t: v for t, v in answers.items() if v != base(t)}

    # Name-level rewrite for VODs without frames: only when every template of an old name
    # got the same answer (then every reading of that name was one of those pictures).
    by_base: dict[str, set[str]] = {}
    for s in stems:
        by_base.setdefault(base(s), set()).add(s)
    rewrite: dict[str, str | None] = {}
    partial = []
    for old in {base(t) for t in answers}:
        new = {answers.get(s, old) for s in by_base[old]}
        if len(new) == 1:
            v = new.pop()
            rewrite[old] = None if v == NONE else v      # _unknown stays a template id below
        else:
            partial.append(old)

    used = [int(s[5:]) for s in stems if s.startswith("item_") and s[5:].isdigit()]
    nxt = max(used, default=0) + 1
    taken = set(stems)
    plan = []
    for t, v in sorted(answers.items()):
        if v == NONE:
            plan.append((t, None))
            continue
        if v == UNKNOWN:
            new = f"item_{nxt:03d}"
            nxt += 1
        else:
            new, k = v, 2
            while new in taken:
                new, k = f"{v}~{k}", k + 1
        taken.discard(t)
        taken.add(new)
        plan.append((t, new))
        if rewrite.get(base(t)) == UNKNOWN:
            rewrite[base(t)] = new
    for t, new in plan:
        print(f"  {t:<34} -> {new or 'removed (not an item)'}")
    print(f"readings rewritten by name in VODs without frames: {rewrite or 'none'}")
    if partial:
        print(f"not fixable by name there (other templates of the name are right): {sorted(partial)}")
    if args.dry_run:
        return

    for t, new in plan:
        src = SEEN / f"{t}.png"
        if new is None:
            REMOVED.mkdir(exist_ok=True)
            shutil.move(str(src), REMOVED / src.name)
        else:
            src.rename(SEEN / f"{new}.png")
    if args.no_finish:
        return

    full, slim = [], []
    for d in sorted(p for p in paths.STATES.iterdir() if p.is_dir() and list(p.glob("game_??.jsonl"))):
        (full if list((paths.FRAMES / d.name).glob("game_??")) else slim).append(d.name)
    touched = list(full)
    for v in full:
        run([sys.executable, "scripts/harvest_item_icons.py", "--frames", str(paths.FRAMES / v),
             "--update-states", str(paths.STATES / v), "--no-harvest"])
    if rewrite:
        for v in slim:
            changed = 0
            for f in sorted((paths.STATES / v).glob("game_??.jsonl")):
                rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l]
                for r in rows:
                    bi = r.get("bench_items")
                    if bi and any(n in rewrite for n in bi if n):
                        r["bench_items"] = [rewrite.get(n, n) if n in rewrite else n for n in bi
                                            if not (n in rewrite and rewrite[n] is None)]
                        changed += 1
                f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            print(f"  {v}: {changed} rows rewritten")
            if changed:
                touched.append(v)
    for v in touched:
        states = str(paths.STATES / v)
        run([sys.executable, "scripts/clean_states.py", "--states", states])
        run([sys.executable, "scripts/build_dataset.py", "--states", states])
        run([sys.executable, "scripts/upload_states.py", "--", v])
    print(f"Done: {len(plan)} templates changed, {len(touched)} VODs updated.")


if __name__ == "__main__":
    main()
