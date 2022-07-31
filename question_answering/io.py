import json
import tqdm
import numpy as np
from typing import Dict


def load_squad_v1_data(path: str) -> Dict:
    try:
        json_file = open(path, mode='r', encoding="utf-8")
        return json.load(json_file)
    except IOError:
        raise IOError


def load_glove_embeddings(path: str) -> Dict[str, np.ndarray]:
    glove_embeddings = {}
    try:
        file = open(path, mode='r', encoding="utf-8")
        for line in tqdm.tqdm(file):
            values = line.split(' ')
            glove_embeddings[values[0]] = np.asarray(values[1:], dtype="float32")
        return glove_embeddings
    except IOError:
        raise IOError
