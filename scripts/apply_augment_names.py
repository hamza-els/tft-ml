"""
After labelling augments: name the templates, then re-clean, re-encode and
upload every finished VOD so the names reach the states and datasets.

Usage:
    python scripts/apply_augment_names.py          # every finished VOD (~/tft_data/state_vectors/*/.done)
    python scripts/apply_augment_names.py VOD ...  # only these

Runs scripts/name_augment_templates.py (game labels from label_augments.py and
template names from label_new_augments.py -> assets/templates/augment_names.json),
then per VOD clean_states.py, build_dataset.py and upload_states.py, then
eval_augments.py --tier for the held-out score. VODs a batch is still working
on (no .done) are left alone. Needs no frames: it only reads the state files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import subprocess

from src.utils import paths

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], quiet: bool = True) -> bool:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                            capture_output=quiet, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 and quiet:
        print(result.stdout[-2000:], result.stderr[-2000:])
    return result.returncode == 0


def main():
    vods = sys.argv[1:] or sorted(p.parent.name for p in paths.STATES.glob("*/.done"))
    print("Naming templates ...")
    if not run(["scripts/name_augment_templates.py"], quiet=False):
        sys.exit("name_augment_templates.py failed")
    for vod in vods:
        states = str(paths.STATES / vod)
        ok = (run(["scripts/clean_states.py", "--states", states])
              and run(["scripts/build_dataset.py", "--states", states])
              and run(["scripts/upload_states.py", vod]))
        print(f"  {vod}: {'ok' if ok else 'FAILED'}")
    print("Held-out score:")
    run(["scripts/eval_augments.py", "--tier"], quiet=False)


if __name__ == "__main__":
    main()
