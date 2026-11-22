"""
Extract, split into games and dedup in one streaming pass, without writing
every frame to disk first.

The old first step (run_pipeline.py) wrote every 2-fps frame of the whole
stream as a JPEG (~43k files, ~22 GB for 6 h), read a fifth of them back to
find the games, moved the lobby out, then read every game frame back to drop
~3/4 of them as duplicates. Here:

  1. probe(): one frame every PROBE_SEC seconds, straight from the video
     (OpenCV seeking), to find roughly where games are. About 22% of stream
     time is outside games (9-42% per VOD, 2026-11-20); it is never decoded.
  2. Each stretch with a game (plus margins) is decoded by ffmpeg at 2 fps and
     piped in as raw frames; stretches are done in parallel.
  3. As frames arrive, the old rules run on them unchanged:
       games: segment_games() logic over every 5th frame number (the same
              frames the old scan sampled): a run of stage-icon frames with no
              gap over 90 s, or a return to the stage-1 layout after 5 min of
              normal layout; it ends at its last gold-icon frame; padded one
              scan step each side, never past the midpoint to a neighbour;
              under 10 min dropped.
       dedup: frame_dedup's rules (HUD-aware, combat-aware, board-aware,
              keep-alive every 5 s), restarted at each game's first frame.
     Only kept frames are written, as JPEG, into <out>/game_XX/ with the same
     frame numbers as before (frame N is at (N-1)/2 s).

Differences from the old step, by design small: dedup compares the decoded
frames rather than JPEG re-reads (no compression noise), and a game's start
is found at the first stage-icon frame as it streams past (a 5-frame buffer
gives the old padding). Frames written after a game's end (it is only known
90 s later) are deleted when it closes.

With states_dir (part 2), each game is also read as soon as its last frame is
written, while other stretches are still decoding: states (OCR etc.), the
augment overlay and the units, by the same per-game functions the step
scripts use (extract_states.process_game, read_augment_overlay.read_game,
read_units.process_game), on the JPEGs just written, so the readings are
those of the step scripts and the models see the input they were trained on.
Decoding leaves the GPU idle and ~40% of the CPU free; a reading worker holds
~2.5 GB (EasyOCR + classifiers), so only read_jobs_decoding of them run while
decoding (RAM: 3 decode workers + desktop apps leave ~3.5 GB), read_jobs after.
When every game was read, <states_dir>/.stream_read marks it; process_vods.py
then skips those three steps. The item step (harvest_item_icons.py) still runs
after: it grows the templates in game order across the whole VOD.
"""

import json
import multiprocessing
import queue as queue_mod
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from src.data_collection.frame_dedup import CHANGE_THRESHOLD, COMBAT_REGIONS, _has_changed, board_signature
from src.data_collection.frame_game_splitter import HudSample, segment_games
from src.vision.arena import is_planning
from src.vision.hud import HudDetector

FPS = 2.0
W, H = 1920, 1080
PROBE_SEC = 20.0
STEP = 5                     # scan step in frame numbers (old scan_step_sec 2.5 at 2 fps)
MAX_GAP_SEC, MIN_GAME_SEC, RESTART_SEC = 90.0, 600.0, 300.0
KEEP_GAP_SEC, BOARD_GAP_SEC = 5.0, 2.0
JPEG_QUALITY = 88            # same size as the old extractor's ffmpeg -q:v 2 (~355 KB a frame; 95 was ~575)


def frame_no(t: float) -> int:
    """Frame number of the 2-fps frame at t seconds (ffmpeg's image2 numbering starts at 1)."""
    return int(round(t * FPS)) + 1


def frame_time(n: int) -> float:
    return (n - 1) / FPS


# ── 1. probing ──────────────────────────────────────────────────────────────

def _probe_times(video: str, times: list[float]) -> list[HudSample]:
    detector = HudDetector()
    cap = cv2.VideoCapture(video)
    samples = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok and frame is not None and frame.shape[:2] == (H, W):
            hud = detector.detect(frame)
            samples.append(HudSample(frame_no(t), hud.stage_visible, hud.shop_visible, hud.stage_one_layout))
    cap.release()
    return samples


def probe(video: str, step_sec: float = PROBE_SEC, workers: int = 4) -> tuple[list[HudSample], float]:
    """
    HUD samples every step_sec seconds, and the video's length in seconds.

    A seek decodes forward from the previous keyframe (6 s apart in OGLPl4JkAmQ),
    ~0.8 s a probe there (0.2 s in htwxJLXZE_g): ~13 min for 5 h in one process.
    Nothing else runs yet, so the timeline is split over several processes.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    times, t = [], 0.0
    while t < duration:
        times.append(t)
        t += step_sec
    workers = max(1, min(workers, len(times) // 20 or 1))
    chunks = [times[i * len(times) // workers:(i + 1) * len(times) // workers] for i in range(workers)]
    if workers == 1:
        return _probe_times(video, times), duration
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_probe_times, [video] * workers, chunks))
    return [s for part in parts for s in part], duration


def decode_ranges(samples: list[HudSample], duration: float, step_sec: float = PROBE_SEC) -> list[tuple[float, float]]:
    """Stretches of video (start s, end s) that hold games, with margins, merged where they touch."""
    coarse = segment_games(samples, FPS, MAX_GAP_SEC, 0.0, RESTART_SEC)
    pad = 2 * step_sec + STEP / FPS
    out: list[list[float]] = []
    for start, end in coarse:
        lo = max(0.0, frame_time(start) - pad)
        hi = min(duration, frame_time(end) + pad + MAX_GAP_SEC)       # room to see the run close
        lo = round(lo * 2) / 2                                        # on the 2-fps grid
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


# ── 2-3. one stretch: decode, find games, dedup, write kept frames ──────────

@dataclass
class _Run:
    """A game being read: the old segment_games run plus that folder's dedup state."""
    folder: Path
    start: int                      # first stage-icon sample
    lo: int                         # first frame in the game folder
    end: int | None = None          # last gold-icon sample so far
    last_stage: int = 0
    normal_since: int | None = None
    written: list[int] = field(default_factory=list)
    # dedup state (frame_dedup.deduplicate_frames)
    reference: np.ndarray | None = None
    ref_number: int = 0
    ref_bar: bool | None = None
    ref_board: tuple | None = None
    last_board_keep: int | None = None
    kept_board: int = 0

    def dedup_keep(self, frame: np.ndarray, number: int, hud) -> bool:
        bar, dx = hud.shop_visible, hud.bottom_dx
        planning = is_planning(frame)
        if bar is False and self.ref_bar is False:
            regions = ["stage"]
        elif not planning:
            regions = COMBAT_REGIONS
        else:
            regions = None
        keep = (self.reference is None or bar != self.ref_bar
                or number - self.ref_number >= KEEP_GAP_SEC * FPS
                or _has_changed(frame, self.reference, CHANGE_THRESHOLD, regions, dx))
        board_now = None
        if planning and bar is not False:
            board_now = board_signature(frame)
            if self.ref_board is None:
                self.ref_board = board_now
            elif (not keep and board_now != self.ref_board
                  and (self.last_board_keep is None or number - self.last_board_keep >= BOARD_GAP_SEC * FPS)):
                keep = True
                self.last_board_keep = number
                self.kept_board += 1
        if keep:
            self.reference, self.ref_number, self.ref_bar = frame, number, bar
            self.ref_board = board_now if board_now is not None else (self.ref_board if planning else None)
        return keep


def _ffmpeg_frames(video: str, start: float, end: float):
    """Raw BGR frames at 2 fps from start to end seconds, with their frame numbers."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-i", video,
           "-t", f"{end - start:.3f}", "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)
    size, k = W * H * 3, 0
    first = frame_no(start)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield first + k, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _is_game(start: int, end: int | None) -> bool:
    """The caller's rule for keeping a run as a game (a gold icon seen, at least MIN_GAME_SEC long)."""
    return end is not None and (end - start) / FPS >= MIN_GAME_SEC


def stream_range(video: str, start: float, end: float, out_dir: str, tag: str, done_queue=None) -> list[dict]:
    """
    Decode one stretch and leave its games' kept frames in out_dir/<tag>_run_NN/.
    Returns one dict per run (start, end, lo, hi, folder, kept, board-kept); the caller drops
    short games and numbers the folders. With done_queue, each run that will be kept as a game
    is put on it (its folder) the moment its last frame is written, so it can be read meanwhile.
    """
    out = Path(out_dir)
    detector = HudDetector()
    runs: list[_Run] = []
    closed: list[_Run] = []
    current: _Run | None = None
    recent: list[tuple[int, np.ndarray, object]] = []      # last STEP frames, for the start padding

    def close(run: _Run, next_start: int | None):
        """End a run: trim frames past its end (+ padding, never past the midpoint to the next game)."""
        if run.end is None:
            hi = run.lo - 1                                       # no gold icon ever: nothing to keep
        else:
            hi = run.end + STEP
            if next_start is not None:
                hi = min(hi, (run.end + next_start) // 2)
        for n in [n for n in run.written if n > hi]:
            (run.folder / f"frame_{n:06d}.jpg").unlink(missing_ok=True)
        run.written = [n for n in run.written if n <= hi]
        run.hi = hi
        closed.append(run)
        if done_queue is not None and _is_game(run.start, run.end):
            done_queue.put(str(run.folder))

    for number, frame in _ffmpeg_frames(video, start, end):
        hud = detector.detect(frame)
        sampled = (number - 1) % STEP == 0
        if sampled and hud.stage_visible:
            gap = current is not None and number - current.last_stage > MAX_GAP_SEC * FPS
            restarted = (current is not None and hud.stage_one_layout and current.normal_since is not None
                         and number - current.normal_since > RESTART_SEC * FPS)
            if current is not None and (gap or restarted):
                prev_end = current.end
                close(current, number)
                current = None
                lo_floor = (prev_end + number) // 2 + 1 if prev_end is not None else None
            else:
                lo_floor = None
            if current is None:
                lo = number - STEP
                if lo_floor is not None:
                    lo = max(lo, lo_floor)
                folder = out / f"{tag}_run_{len(runs):02d}"
                folder.mkdir(parents=True, exist_ok=True)
                current = _Run(folder=folder, start=number, lo=lo)
                runs.append(current)
                # frames already streamed past that belong to the game (the padding)
                for n, f, h in recent:
                    if n >= lo and current.dedup_keep(f, n, h):
                        cv2.imwrite(str(folder / f"frame_{n:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        current.written.append(n)
            current.last_stage = number
            if not hud.stage_one_layout and current.normal_since is None:
                current.normal_since = number
            if hud.shop_visible:
                current.end = number
        if current is not None:
            if current.dedup_keep(frame, number, hud):
                cv2.imwrite(str(current.folder / f"frame_{number:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                current.written.append(number)
        recent.append((number, frame.copy(), hud))
        if len(recent) > STEP:
            recent.pop(0)
    if current is not None:
        close(current, None)
    return [{"start": r.start, "end": r.end, "lo": r.lo, "hi": r.hi, "folder": str(r.folder),
             "kept": len(r.written), "board_kept": r.kept_board} for r in closed]


# ── part 2: read a finished game (in a reading worker) ─────────────────────

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def read_run(folder: str, states_dir: str, overlay_frames: int = 25) -> dict:
    """
    Read one game's kept frames the way the step scripts do: states
    (extract_states.py), augment overlay (read_augment_overlay.py), units
    (read_units.py). Writes <states_dir>/<folder name>.jsonl; returns the overlay
    reading and timings. Models load once per worker process (the scripts' globals).
    """
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import extract_states
    import read_augment_overlay
    import read_units
    from src.vision.ocr_pipeline import _get_reader

    frames = Path(folder)
    states = Path(states_dir) / f"{frames.name}.jsonl"
    t0 = time.time()
    states_msg = extract_states.process_game(str(frames), str(states), FPS, None, progress=False)
    t1 = time.time()
    overlay = read_augment_overlay.read_game(states, frames, _get_reader(), overlay_frames)
    t2 = time.time()
    units_msg = read_units.process_game(str(states), str(frames.parent), False, progress=False)
    t3 = time.time()
    return {"run": frames.name, "overlay": overlay, "states_s": round(t1 - t0, 1),
            "overlay_s": round(t2 - t1, 1), "units_s": round(t3 - t2, 1),
            "msg": f"{states_msg.strip()} | {units_msg.strip()}"}


# ── whole video ─────────────────────────────────────────────────────────────

STREAM_READ_MARK = ".stream_read"


def stream_extract(video: str, out_dir: str, workers: int = 3, log=print, states_dir: str | None = None,
                   read_jobs: int = 2, read_jobs_decoding: int = 1, probe_workers: int = 4) -> dict:
    """
    Probe, decode the game stretches in parallel, and leave <out_dir>/game_XX/ with kept frames.
    With states_dir, also read every game into <states_dir>/game_XX.jsonl (+ augment_overlay.json)
    as it finishes (see the module docstring). Returns timings and per-game counts.
    """
    t0 = time.time()
    samples, duration = probe(video, workers=probe_workers)
    ranges = decode_ranges(samples, duration)
    t_probe = time.time() - t0
    covered = sum(b - a for a, b in ranges)
    log(f"  probed {len(samples)} frames in {t_probe:.0f} s; decoding {len(ranges)} stretch(es), "
        f"{covered / 3600:.1f} of {duration / 3600:.1f} h")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reading = states_dir is not None
    sdir = Path(states_dir) if reading else None
    if reading:
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / STREAM_READ_MARK).unlink(missing_ok=True)
        for stale in [*sdir.glob("s??_run_*.jsonl"), *sdir.glob("s??_run_*.jsonl.part")]:
            stale.unlink()
        # state files of an earlier attempt belong to frames that are being replaced
        old = sorted(sdir.glob("game_??.jsonl")) + sorted(sdir.glob("game_??.clean.jsonl"))
        if old:
            aside = sdir / f"replaced_{time.strftime('%Y%m%d_%H%M%S')}"
            aside.mkdir()
            for p in old:
                p.replace(aside / p.name)
            log(f"  moved {len(old)} earlier state file(s) to {aside}")

    t1 = time.time()
    manager = multiprocessing.Manager() if reading else None
    done_q = manager.Queue() if reading else None
    n_decode = max(1, min(workers, len(ranges)))
    decode_pool = ProcessPoolExecutor(max_workers=n_decode)
    read_pool = ProcessPoolExecutor(max_workers=max(read_jobs, read_jobs_decoding)) if reading else None
    reads: dict = {}                       # run name -> result dict, or an exception
    t_decode = t_read_tail = None
    try:
        dfuts = [decode_pool.submit(stream_range, video, a, b, str(out), f"s{i:02d}", done_q)
                 for i, (a, b) in enumerate(ranges)]
        if not reading:
            parts = [f.result() for f in dfuts]
            t_decode = time.time() - t1
        else:
            waiting: list[str] = []
            running: dict = {}             # future -> run folder
            while True:
                decoding = not all(f.done() for f in dfuts)
                if not decoding and t_decode is None:
                    t_decode = time.time() - t1
                while True:
                    try:
                        waiting.append(done_q.get_nowait())
                    except queue_mod.Empty:
                        break
                for f in [f for f in running if f.done()]:
                    name = Path(running.pop(f)).name
                    try:
                        r = f.result()
                        reads[name] = r
                        log(f"  read {name}: states {r['states_s']:.0f} s, overlay {r['overlay_s']:.0f} s, "
                            f"units {r['units_s']:.0f} s")
                    except Exception as e:      # noqa: BLE001 - left to the step scripts (no mark)
                        reads[name] = e
                        log(f"  reading {name} FAILED: {e!r} (the step scripts will read it)")
                limit = read_jobs_decoding if decoding else read_jobs
                while waiting and len(running) < limit:
                    folder = waiting.pop(0)
                    running[read_pool.submit(read_run, folder, str(sdir))] = folder
                if not decoding and not waiting and not running:
                    break
                time.sleep(1.0)
            t_read_tail = time.time() - t1 - t_decode
            parts = [f.result() for f in dfuts]
    finally:
        decode_pool.shutdown(wait=True)
        if read_pool is not None:
            read_pool.shutdown(wait=True)
        if manager is not None:
            manager.shutdown()

    runs = sorted((r for part in parts for r in part), key=lambda r: r["start"])
    games, overlay = [], {}
    for r in runs:
        folder = Path(r["folder"])
        if not _is_game(r["start"], r["end"]):
            shutil.rmtree(folder, ignore_errors=True)
            continue
        name = f"game_{len(games):02d}"
        dest = out / name
        if dest.exists():
            shutil.rmtree(dest)
        folder.rename(dest)
        if reading and isinstance(reads.get(folder.name), dict):
            (sdir / f"{folder.name}.jsonl").replace(sdir / f"{name}.jsonl")
            overlay[name] = reads[folder.name]["overlay"]
        games.append({**r, "folder": str(dest)})
        log(f"  {name}: frames {r['lo']}-{r['hi']} "
            f"({frame_time(r['start']) / 60:.1f}-{frame_time(r['end']) / 60:.1f} min), {r['kept']} kept "
            f"({r['board_kept']} for a board change)")
    for leftover in out.glob("s??_run_*"):
        shutil.rmtree(leftover, ignore_errors=True)
    read_ok = reading and bool(games) and len(overlay) == len(games)
    if reading:
        for stale in [*sdir.glob("s??_run_*.jsonl"), *sdir.glob("s??_run_*.jsonl.part")]:
            stale.unlink()
        if read_ok:
            (sdir / "augment_overlay.json").write_text(json.dumps(overlay, indent=1), encoding="utf-8")
            (sdir / STREAM_READ_MARK).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        else:
            log(f"  read {len(overlay)} of {len(games)} games while streaming: the step scripts read the rest")
    summary = {"duration_h": round(duration / 3600, 2), "decoded_h": round(covered / 3600, 2),
               "probe_s": round(t_probe, 1), "decode_s": round(t_decode or 0.0, 1),
               "read_tail_s": round(t_read_tail, 1) if t_read_tail is not None else None,
               "read": read_ok, "games": games,
               "reads": {k: {x: v[x] for x in ("states_s", "overlay_s", "units_s")}
                         for k, v in reads.items() if isinstance(v, dict)}}
    (out / "stream_extract.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
