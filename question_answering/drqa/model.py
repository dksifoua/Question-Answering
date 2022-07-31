from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import AlignedQuestionEmbeddingLayer, BiLinearAttentionLayer, StackedBiLSTMsLayer, QuestionEncodingLayer


class DrQA(nn.Module):
    
    def __init__(self, vocabulary_size: int, embedding_size, n_extra_features: int, hidden_size: int, n_layers: int,
                 dropout: float, padding_index: int):
        super(DrQA, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.n_extra_features = n_extra_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.padding_index = padding_index

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_index
        )
        self.aligned_question_embedding_layer = AlignedQuestionEmbeddingLayer(
            embedding_size=embedding_size,
            hidden_size=hidden_size
        )
        self.context_lstm_layer = StackedBiLSTMsLayer(
            embedding_size=embedding_size * 2 + n_extra_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout
        )
        self.question_encoding_layer = QuestionEncodingLayer(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            n_layers=n_layers
        )
        self.bilinear_attention_start_layer = BiLinearAttentionLayer(
            context_hidden_size=hidden_size * n_layers * 2,
            question_hidden_size=hidden_size * n_layers * 2
        )
        self.bilinear_attention_end_layer = BiLinearAttentionLayer(
            context_hidden_size=hidden_size * n_layers * 2,
            question_hidden_size=hidden_size * n_layers * 2
        )

    def make_sequence_mask(self, input_sequence: Tensor) -> Tensor:
        return input_sequence != self.padding_index

    @staticmethod
    def decode(starts: Tensor, ends: Tensor):
        pass

    def forward(self, context_sequence: Tensor, context_lengths: Tensor, question_sequence: Tensor,
                question_lengths: Tensor, exact_matches: Tensor, part_of_speeches: Tensor, named_entity_types: Tensor,
                normalized_term_frequencies: Tensor) -> Tuple[Tensor, Tensor]:
        context_mask = self.make_sequence_mask(input_sequence=context_sequence)
        question_mask = self.make_sequence_mask(input_sequence=question_sequence)
