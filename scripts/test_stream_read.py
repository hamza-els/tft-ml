"""
Streaming with reading (stream_extract with states_dir) vs streaming then the
step scripts, on the same clip: same kept frames, same state rows, same
overlay, and the time each takes.

Usage:
    python scripts/test_stream_read.py CLIP.mp4 [--jobs 2]

Writes into <clip folder>/A_* (old: stream, then extract_states.py and
read_units.py with --jobs, overlay in-process as read_augment_overlay.py does)
and <clip folder>/B_* (new: one pass). The item step is left out of both: it
runs after either way, unchanged, and writes the repo's item templates.
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

from src.data_collection.stream_extract import stream_extract

SCRIPTS = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--skip-old", action="store_true", help="Reuse A_* from an earlier run")
    args = parser.parse_args()
    clip = Path(args.clip).expanduser().resolve()
    root = clip.parent
    a_frames, a_data = root / "A_frames", root / "A_data"
    b_frames, b_states = root / "B_frames", root / "B_states"
    a_states = a_data / "state_vectors" / a_frames.name
    env = {**os.environ, "TFT_DATA": str(a_data), "PYTHONIOENCODING": "utf-8"}

    times = {}
    if not args.skip_old:
        for p in (a_frames, a_data):
            shutil.rmtree(p, ignore_errors=True)
        t0 = time.time()
        stream_extract(str(clip), str(a_frames))
        times["A stream"] = time.time() - t0
        t0 = time.time()
        subprocess.run([sys.executable, str(SCRIPTS / "extract_states.py"), "--frames", str(a_frames),
                        "--jobs", str(args.jobs)], check=True, env=env)
        times["A read states"] = time.time() - t0
        t0 = time.time()
        sys.path.insert(0, str(SCRIPTS))
        import read_augment_overlay
        from src.vision.ocr_pipeline import _get_reader
        reader = _get_reader()
        overlay = {s.stem: read_augment_overlay.read_game(s, a_frames / s.stem, reader, 25)
                   for s in sorted(a_states.glob("game_??.jsonl"))}
        (a_states / "augment_overlay.json").write_text(json.dumps(overlay, indent=1), encoding="utf-8")
        times["A overlay"] = time.time() - t0
        t0 = time.time()
        subprocess.run([sys.executable, str(SCRIPTS / "read_units.py"), "--frames", str(a_frames),
                        "--states", str(a_states), "--jobs", str(args.jobs)], check=True, env=env)
        times["A read units"] = time.time() - t0
        (root / "A_times.json").write_text(json.dumps(times, indent=1), encoding="utf-8")
    else:
        times = json.loads((root / "A_times.json").read_text(encoding="utf-8"))

    for p in (b_frames, b_states):
        shutil.rmtree(p, ignore_errors=True)
    t0 = time.time()
    summary = stream_extract(str(clip), str(b_frames), states_dir=str(b_states), read_jobs=args.jobs)
    times["B stream + read"] = time.time() - t0

    old = sum(v for k, v in times.items() if k.startswith("A"))
    print("\nTIMES")
    for k, v in times.items():
        print(f"  {k:18s} {v / 60:6.1f} min")
    print(f"  old total {old / 60:.1f} min, new {times['B stream + read'] / 60:.1f} min "
          f"(decode {summary['decode_s'] / 60:.1f}, reading after decode {(summary['read_tail_s'] or 0) / 60:.1f}); "
          f"read in the pass: {summary['read']}")

    print("\nFRAMES AND STATES")
    for g in sorted(p.name for p in a_frames.glob("game_*")):
        fa = {p.name for p in (a_frames / g).glob("*.jpg")}
        fb = {p.name for p in (b_frames / g).glob("*.jpg")}
        ra, rb = rows(a_states / f"{g}.jsonl"), rows(b_states / f"{g}.jsonl") if (b_states / f"{g}.jsonl").exists() else []
        diff_rows, diff_fields = 0, {}
        for x, y in zip(ra, rb):
            if x != y:
                diff_rows += 1
                for k in set(x) | set(y):
                    if x.get(k) != y.get(k):
                        diff_fields[k] = diff_fields.get(k, 0) + 1
        units_a = sum(1 for r in ra if r.get("units_raw") is not None)
        units_b = sum(1 for r in rb if r.get("units_raw") is not None)
        print(f"  {g}: frames A {len(fa)} B {len(fb)} same {len(fa & fb)}; rows A {len(ra)} B {len(rb)}, "
              f"{diff_rows} differ {dict(sorted(diff_fields.items(), key=lambda kv: -kv[1])[:6])}; "
              f"units read A {units_a} B {units_b}")
    oa = json.loads((a_states / "augment_overlay.json").read_text(encoding="utf-8"))
    ob_path = b_states / "augment_overlay.json"
    ob = json.loads(ob_path.read_text(encoding="utf-8")) if ob_path.exists() else {}
    same = sum(1 for g in oa if ob.get(g, {}).get("augments") == oa[g]["augments"])
    print(f"  overlay: same augments in {same} of {len(oa)} games; marker written: "
          f"{(b_states / '.stream_read').exists()}")


if __name__ == "__main__":
    main()
