from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to a YAML config.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict at top-level, got: {type(config)}")

    return config
