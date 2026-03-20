import yaml
import os
from typing import Any


def getFromConfig(configPath: str, key: str) -> Any | None:
    if not os.path.exists(configPath):
        return None

    with open(configPath, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config.get(key)
