from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_configs(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load all project YAML files and return them by filename stem."""
    config_dir = Path(config_dir)
    names = ("dataset", "path", "wandb", "model", "train")
    return {name: load_yaml(config_dir / f"{name}.yaml") for name in names}


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
