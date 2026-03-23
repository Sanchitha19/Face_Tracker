import json
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_config():
    """
    Reads config.json from the project root.
    Caches the result after first load.
    Validates that required top-level keys exist.
    """
    # Current script path should be in src, go one level up for root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Required top-level keys
    required_keys = [
        "video_source", "skip_frames", "detection", 
        "recognition", "tracking", "logging", 
        "database", "display"
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    return config
