"""
After labelling augments (or once new videos have overlay names): rebuild the
augment templates from every named game, read the augment bars again with
them, and re-clean, re-encode and upload every finished VOD.

Usage:
    python scripts/apply_augment_names.py           # every finished VOD (~/tft_data/state_vectors/*/.done)
    python scripts/apply_augment_names.py VOD ...   # re-read and re-clean only these (templates still rebuilt)
    python scripts/apply_augment_names.py --eval    # also the held-out test (slower)

Steps:
  1. scripts/build_augment_templates.py --write: medoid templates per augment from
     games named by Hamza's label or the stream overlay, same-icon names merged
     (assets/templates/augments_v2/: names.json, aliases.json, groups.json).
  2. per VOD with frames on disk (full or augment-only copies):
     backfill_states.py --augments-only, so every frame's reading uses those templates.
  3. per VOD: clean_states.py (label > overlay > reader), build_dataset.py, upload_states.py.
VODs a batch is still working on (no .done) are left alone.
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


def frames_for(vod: str) -> Path | None:
    for root in (paths.FRAMES, paths.AUG_FRAMES):
        if (root / vod).exists():
            return root / vod
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    evaluate = "--eval" in sys.argv
    vods = args or sorted(p.parent.name for p in paths.STATES.glob("*/.done"))
    print("Building templates from named games ...")
    if not run(["scripts/build_augment_templates.py", "--write"] + (["--eval"] if evaluate else []), quiet=False):
        sys.exit("build_augment_templates.py failed")
    for vod in vods:
        states = str(paths.STATES / vod)
        frames = frames_for(vod)
        reread = frames is None or run(["scripts/backfill_states.py", "--augments-only",
                                         "--frames", str(frames), "--states", states])
        ok = (reread and run(["scripts/clean_states.py", "--states", states])
              and run(["scripts/build_dataset.py", "--states", states])
              and run(["scripts/upload_states.py", "--", vod]))
        note = "" if frames else " (no frames on disk: augments not re-read)"
        print(f"  {vod}: {'ok' if ok else 'FAILED'}{note}", flush=True)


if __name__ == "__main__":
    main()
