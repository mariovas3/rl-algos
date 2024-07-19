import json
from pathlib import Path


def save_to_json(obj, filepath: Path, **kwargs):
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(obj, file, **kwargs)
