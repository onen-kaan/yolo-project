import yaml
import os
from typing import Any


def get_from_config(configPath: str, key: str) -> Any:
    if not os.path.exists(configPath):
        return None

    with open(configPath, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config.get(key)
