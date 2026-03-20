import yaml
import os
from typing import Any


def get_from_config(config_path: str, key: str) -> Any | None:
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config.get(key)

