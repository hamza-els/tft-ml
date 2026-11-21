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
"""

import json
import shutil
import subprocess
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

def probe(video: str, detector: HudDetector, step_sec: float = PROBE_SEC) -> tuple[list[HudSample], float]:
    """HUD samples every step_sec seconds, and the video's length in seconds."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    samples, t = [], 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok and frame is not None and frame.shape[:2] == (H, W):
            hud = detector.detect(frame)
            samples.append(HudSample(frame_no(t), hud.stage_visible, hud.shop_visible, hud.stage_one_layout))
        t += step_sec
    cap.release()
    return samples, duration


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


def stream_range(video: str, start: float, end: float, out_dir: str, tag: str) -> list[dict]:
    """
    Decode one stretch and leave its games' kept frames in out_dir/<tag>_run_NN/.
    Returns one dict per run (start, end, lo, hi, folder, kept, board-kept); the caller drops
    short games and numbers the folders.
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


# ── whole video ─────────────────────────────────────────────────────────────

def stream_extract(video: str, out_dir: str, workers: int = 3, log=print) -> dict:
    """
    Probe, decode the game stretches in parallel, and leave <out_dir>/game_XX/ with kept frames.
    Returns timings and per-game counts.
    """
    t0 = time.time()
    detector = HudDetector()
    samples, duration = probe(video, detector)
    ranges = decode_ranges(samples, duration)
    t_probe = time.time() - t0
    covered = sum(b - a for a, b in ranges)
    log(f"  probed {len(samples)} frames in {t_probe:.0f} s; decoding {len(ranges)} stretch(es), "
        f"{covered / 3600:.1f} of {duration / 3600:.1f} h")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(ranges)))) as pool:
        parts = list(pool.map(stream_range, [video] * len(ranges), [a for a, _ in ranges],
                              [b for _, b in ranges], [str(out)] * len(ranges),
                              [f"s{i:02d}" for i in range(len(ranges))]))
    t_decode = time.time() - t1
    runs = sorted((r for part in parts for r in part), key=lambda r: r["start"])
    games = []
    for r in runs:
        folder = Path(r["folder"])
        too_short = r["end"] is None or (r["end"] - r["start"]) / FPS < MIN_GAME_SEC
        if too_short:
            shutil.rmtree(folder, ignore_errors=True)
            continue
        dest = out / f"game_{len(games):02d}"
        if dest.exists():
            shutil.rmtree(dest)
        folder.rename(dest)
        games.append({**r, "folder": str(dest)})
        log(f"  game_{len(games) - 1:02d}: frames {r['lo']}-{r['hi']} "
            f"({frame_time(r['start']) / 60:.1f}-{frame_time(r['end']) / 60:.1f} min), {r['kept']} kept "
            f"({r['board_kept']} for a board change)")
    for leftover in out.glob("s??_run_*"):
        shutil.rmtree(leftover, ignore_errors=True)
    summary = {"duration_h": round(duration / 3600, 2), "decoded_h": round(covered / 3600, 2),
               "probe_s": round(t_probe, 1), "decode_s": round(t_decode, 1), "games": games}
    (out / "stream_extract.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
