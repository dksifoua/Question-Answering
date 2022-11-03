import torch
import torch.nn as nn


class CharacterEmbeddingLayer(nn.Module):

    def __init__(self, vocabulary_size: int, embedding_size: int, kernel_size: int, padding_index: int):
        super(CharacterEmbeddingLayer, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.padding_index = padding_index

    def forward(self, character_sequences: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
