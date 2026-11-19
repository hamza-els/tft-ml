"""
Check YouTube VODs before downloading them: grab 20 evenly spaced frames from
each (one ~6 s HLS segment per frame, a few MB, no full download), run the
HUD detector (stage icon = TFT in the calibrated layout), and write a 5x4
contact sheet (green border = stage icon found) plus one JSON line per video.

Usage:
    python scripts/sample_vods.py OUT_DIR VIDEO_ID [VIDEO_ID ...]

The stage count says how much is TFT; look at the sheets for what it can't
tell: other games, and other players' games (watch parties, bootcamp days:
another webcam or a streamer's name/avatar top left), which also show a TFT
HUD. List a channel with:
    python -m yt_dlp --flat-playlist --print "%(id)s|%(duration_string)s|%(title)s" URL/videos
Each video costs one metadata lookup; YouTube's bot check came back after ~45.
Results are cached per video in OUT_DIR, so a re-run skips finished ones.
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2
import imageio_ffmpeg
import numpy as np
import yt_dlp

from src.vision.hud import HudDetector

OUT = Path(sys.argv[1])
IDS = sys.argv[2:]
N = 20
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
det = HudDetector()
OUT.mkdir(parents=True, exist_ok=True)

ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}},
                        "format": "bestvideo[height=1080][protocol=m3u8_native]"})




def segments(url, headers):
    req = urllib.request.Request(url, headers=headers)
    text = urllib.request.urlopen(req, timeout=60).read().decode()
    segs, t, dur = [], 0.0, None
    init = b""
    for line in text.splitlines():
        if line.startswith("#EXT-X-MAP:URI="):
            u = line.split('URI="')[1].split('"')[0]
            init = urllib.request.urlopen(urllib.request.Request(u, headers=headers), timeout=60).read()
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            segs.append((t, dur, line))
            t += dur
            dur = None
    return init, segs


def grab(segs, headers, t):
    init, segs = segs
    for start, dur, seg in segs:
        if start <= t < start + dur:
            break
    data = urllib.request.urlopen(urllib.request.Request(seg, headers=headers), timeout=60).read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(init + data)
        name = f.name
    try:
        r = subprocess.run([FFMPEG, "-v", "error", "-i", name, "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=60)
    finally:
        Path(name).unlink()
    if not r.stdout:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


for vid in IDS:
    res_path = OUT / f"{vid}.json"
    if res_path.exists():
        print(res_path.read_text(), flush=True)
        continue
    try:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    except Exception as e:
        print(json.dumps({"id": vid, "error": str(e)[:200]}), flush=True)
        if "not a bot" in str(e):
            break
        continue
    headers = info.get("http_headers", {})
    segs = segments(info["url"], headers)
    dur = segs[1][-1][0] + segs[1][-1][1]
    times = [dur * (i + 0.5) / N for i in range(N)]

    def safe(t):
        try:
            return grab(segs, headers, t)
        except Exception:
            return None
    with ThreadPoolExecutor(5) as pool:
        frames = list(pool.map(safe, times))
    stage = gold = got = 0
    tiles = []
    for t, f in zip(times, frames):
        if f is None:
            tiles.append(np.zeros((216, 384, 3), np.uint8))
            continue
        got += 1
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        h = det.detect(f)
        stage += h.stage_visible
        gold += h.shop_visible
        tile = cv2.resize(f, (384, 216))
        col = (0, 255, 0) if h.stage_visible else (0, 0, 255)
        cv2.rectangle(tile, (0, 0), (383, 215), col, 3)
        cv2.putText(tile, f"{int(t // 3600)}:{int(t % 3600 // 60):02d}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, N, 5)]
    cv2.imwrite(str(OUT / f"{vid}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    res = {"id": vid, "dur_h": round(dur / 3600, 2), "frames": got, "stage": stage, "gold": gold,
           "height": info.get("height"), "title": info.get("title", "")[:70]}
    res_path.write_text(json.dumps(res))
    print(json.dumps(res), flush=True)
