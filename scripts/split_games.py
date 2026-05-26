"""
CLI: scan a stream VOD, detect game boundaries, and clip each game.

Usage:
    python scripts/split_games.py --video data/raw/vods/mystream.mp4

Clipped games are saved to data/raw/vods/games/.
After clipping, feed each game clip into extract_frames.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import yaml

from src.data_collection.game_splitter import clip_games, find_game_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the full stream .mp4")
    parser.add_argument("--output-dir", default="data/raw/vods/games")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--min-game-minutes", type=float, default=15.0)
    parser.add_argument("--enter-debounce", type=float, default=30.0,
                        help="Seconds of active HUD required to declare game start")
    parser.add_argument("--exit-debounce", type=float, default=120.0,
                        help="Seconds of inactive HUD required to declare game end "
                             "(longer = tolerates alt-tabs without splitting)")
    args = parser.parse_args()

    with open("configs/vision_config.yaml") as f:
        cfg = yaml.safe_load(f)

    regions = cfg["ocr"]["regions"]
    hud_regions = {
        "gold": tuple(regions["gold"]),
        "hp":   tuple(regions["hp"]),
    }

    segments = find_game_segments(
        video_path=args.video,
        hud_regions=hud_regions,
        sample_interval_sec=args.sample_interval,
        min_game_duration_sec=args.min_game_minutes * 60,
        enter_debounce_sec=args.enter_debounce,
        exit_debounce_sec=args.exit_debounce,
    )

    if not segments:
        print("No games detected. Check that hud_regions are calibrated correctly.")
        return

    outputs = clip_games(args.video, segments, args.output_dir)
    print(f"\nClipped {len(outputs)} game(s) to {args.output_dir}/")
    print("Next step: python scripts/extract_frames.py --video-dir", args.output_dir)


if __name__ == "__main__":
    main()
