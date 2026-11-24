"""
VOD download command (scripts/run_pipeline.download): video only, no audio
merge (2026-11-24). Nothing is downloaded: subprocess.run is replaced.

Usage:
    python -m pytest tests/test_data_collection/test_download.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import scripts.download_vods as download_vods
import scripts.run_pipeline as rp


class _Run:
    """Stands in for subprocess.run: records the command, 'downloads' by writing the output file."""
    def __init__(self, returncode=0):
        self.cmds, self.returncode = [], returncode

    def __call__(self, cmd, *a, **k):
        self.cmds.append(cmd)
        if self.returncode == 0:
            out = Path(cmd[cmd.index("--output") + 1].replace("%(id)s", "abc123XYZ_-"))
            out.write_bytes(b"video")
        return type("Done", (), {"returncode": self.returncode})()


def _format(cmd):
    return cmd[cmd.index("--format") + 1]


@pytest.mark.parametrize("fmt", [rp.FORMAT, download_vods.FORMAT])
def test_format_is_video_only(fmt):
    # every alternative of the selector: no audio stream, so no merge
    assert "audio" not in fmt
    for alt in fmt.split("/"):
        assert "+" not in alt
        assert "height<=1080" in alt
    assert fmt.split("/")[0].startswith("bestvideo") and "vcodec^=avc" in fmt.split("/")[0]


def test_download_asks_for_video_only(tmp_path, monkeypatch):
    run = _Run()
    monkeypatch.setattr(rp.subprocess, "run", run)
    out = rp.download("https://www.youtube.com/watch?v=abc123XYZ_-", tmp_path, "ffmpeg", 4)
    assert out == tmp_path / "abc123XYZ_-.mp4" and out.exists()
    (cmd,) = run.cmds
    assert _format(cmd) == rp.FORMAT
    assert cmd[cmd.index("--js-runtimes") + 1] == "node"


def test_existing_file_is_not_downloaded_again(tmp_path, monkeypatch):
    (tmp_path / "abc123XYZ_-.mp4").write_bytes(b"video")
    run = _Run()
    monkeypatch.setattr(rp.subprocess, "run", run)
    assert rp.download("https://youtu.be/abc123XYZ_-", tmp_path, "ffmpeg", 4) == tmp_path / "abc123XYZ_-.mp4"
    assert run.cmds == []


def test_failed_download_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(rp.subprocess, "run", _Run(returncode=1))
    assert rp.download("https://www.youtube.com/watch?v=abc123XYZ_-", tmp_path, "ffmpeg", 4) is None
