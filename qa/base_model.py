import functools

import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def make_sequence_mask(self, input_sequence: Tensor) -> Tensor:
        return input_sequence != self.padding_index

    def load_word_embeddings(self, embedding_matrix: np.ndarray, tune: bool, found_indexes: List[int]):
        def tune_embeddings(grad, word_indexes: List[int]):
            grad[word_indexes] = 0
            return grad

        self.embedding_layer.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        if tune:
            self.embedding_layer.weight.register_hook(
                functools.partial(tune_embeddings, word_indexes=found_indexes)
            )  # Only fine-tune the words that aren't present in embeddings matrix

    @staticmethod
    def decode(starts: Tensor, ends: Tensor) -> Tuple[List[int], List[int], List[float]]:
        """
        :param starts: FloatTensor[batch_size, ctx_seq_len]
        :param ends: FloatTensor[batch_size, ctx_seq_len]
        :return: Tuple[List[int], List[int], List[float]]
        """
        start_indexes, end_indexes, predicted_probabilities = [], [], []
        for i in range(starts.size(0)):
            probabilities = torch.ger(starts[i], ends[i])  # [ctx_seq_len, ctx_seq_len]
            probability, index = torch.topk(probabilities.view(-1), k=1)

            start_indexes.append(index.tolist()[0] // probabilities.size(0))
            end_indexes.append(index.tolist()[0] % probabilities.size(1))

            predicted_probabilities.append(probability.tolist()[0])

        return start_indexes, end_indexes, predicted_probabilities

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
