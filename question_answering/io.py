import json
import tqdm
import pickle
import numpy as np
from typing import Any, Dict


class IO:

    @staticmethod
    def load_from_json(path: str) -> Dict:
        try:
            with open(path, mode='r', encoding="utf-8") as json_file:
                return json.load(json_file)
        except IOError:
            raise IOError

    @staticmethod
    def load_glove_embeddings(path: str) -> Dict[str, np.ndarray]:
        try:
            with open(path, mode='r', encoding="utf-8") as file:
                glove_embeddings = {}
                for line in tqdm.tqdm(file):
                    values = line.split(' ')
                    glove_embeddings[values[0]] = np.asarray(values[1:], dtype="float32")
                return glove_embeddings
        except IOError:
            raise IOError

    @staticmethod
    def save_to_pickle(data: Any, path: str) -> None:
        try:
            with open(path, mode="wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError:
            raise IOError

    @staticmethod
    def load_from_pickle(path: str) -> Any:
        try:
            with open(path, mode="rb") as file:
                return pickle.load(file)
        except IOError:
            raise IOError
