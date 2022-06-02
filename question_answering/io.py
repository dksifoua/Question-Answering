import json
from typing import Dict


def load_squad_v1_data(path: str) -> Dict:
    try:
        json_file = open(path, mode='r', encoding="utf-8")
        return json.load(json_file)
    except IOError:
        raise IOError
