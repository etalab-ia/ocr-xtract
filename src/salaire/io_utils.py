import json
from pathlib import Path


def load_names(names_file_path: Path):
    with open(names_file_path) as filo:
        return json.load(filo)
