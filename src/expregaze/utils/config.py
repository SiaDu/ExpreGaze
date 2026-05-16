from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a yaml config file.

    Example:
        config = load_config("configs/runs/text_main_tt0032138.yaml")
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def get_project_root(config: Dict[str, Any]) -> Path:
    """
    Get project root from config.
    """
    root = config.get("project", {}).get("root")

    if root is None:
        raise KeyError("Missing config field: project.root")

    return Path(root)