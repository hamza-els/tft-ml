"""
Load project-wide config from configs/vision_config.yaml and model_config.yaml.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


vision_cfg = load_config("vision_config")
model_cfg = load_config("model_config")
