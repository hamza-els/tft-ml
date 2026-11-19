"""
Where data lives.

The repo sits in a Google Drive folder, and long jobs writing there have
failed (Drive locks files while it syncs them, and the drive has dropped out
mid-run). So all pipeline work happens on the local disk, under LOCAL
(default ~/tft_data, override with the TFT_DATA environment variable), next
to the frames and videos (~/tft_frames_temp, ~/tft_vods_temp). Only finished
state vectors are copied into the repo (REPO_STATES), by the last step of
scripts/process_vods.py or by scripts/upload_states.py.

    LOCAL/state_vectors/<vod>/game_XX.jsonl     working copy of every state file
    LOCAL/unit_crops/images/                    unit crops (labels and index stay in the repo)
    LOCAL/unit_crops/removed/                   crops taken out by prune_unit_crops.py
    LOCAL/datasets/states/<vod>/game_XX.npz     encoded datasets
    LOCAL/logs/                                 batch logs
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get("TFT_DATA", Path.home() / "tft_data"))

STATES = LOCAL / "state_vectors"
CROP_IMAGES = LOCAL / "unit_crops" / "images"
CROP_REMOVED = LOCAL / "unit_crops" / "removed"
CROP_SCORE_CACHE = LOCAL / "unit_crops" / "confidence_cache.json"
DATASETS = LOCAL / "datasets" / "states"
LOGS = LOCAL / "logs"
ITEM_IMAGES = LOCAL / "item_crops" / "images"
ITEM_CONTEXT = LOCAL / "item_crops" / "context"     # the unit's bar and item row around each icon

# In the repo (on Drive): small files that belong in git.
CROP_DIR = REPO / "data" / "processed" / "unit_crops"     # index.json, labels.json, removed/labels.json
ITEM_DIR = REPO / "data" / "processed" / "item_crops"     # index.json, labels.json
REPO_STATES = REPO / "data" / "processed" / "state_vectors"
