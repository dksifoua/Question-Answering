from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from qa.base_model import BaseModel
from qa.domain import DrQATensorDatasetBatch
from .layers import AlignedQuestionEmbeddingLayer, BiLinearAttentionLayer, StackedBiLSTMsLayer, QuestionEncodingLayer


class DrQA(BaseModel):
    
    def __init__(self, vocabulary_size: int, embedding_size, n_extra_features: int, hidden_size: int, n_layers: int,
                 dropout: float, padding_index: int):
        """
        During prediction, we choose the best span from token $i$ to token $i'$ such that $i ≤ i' ≤ i + 15$ and
        $P_{start}(i)×P_{end}(i')$ is maximized. To make score compatible across paragraphs in one or several retrieved
        documents, we use the un-normalized exponential and take argmax over all considered paragraph spans for our
        final prediction.

        :param vocabulary_size:
        :param embedding_size:
        :param n_extra_features:
        :param hidden_size:
        :param n_layers:
        :param dropout:
        :param padding_index:
        """
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
            embedding_size=embedding_size + hidden_size + n_extra_features,
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

    def forward(self, batch: DrQATensorDatasetBatch) -> Tuple[Tensor, Tensor]:
        """
        :param batch: DrQATensorDatasetBatch
        :return: Tuple[FloatTensor[batch_size, ctx_seq_len], FloatTensor[batch_size, ctx_seq_len]]
        """
        context_sequence = batch.context[0]  # [batch_size, ctx_seq_len]
        context_lengths = batch.context[1]  # [batch_size,]
        question_sequence = batch.question[0]  # [batch_size, qst_seq_len]
        question_lengths = batch.question[1]  # [batch_size,]
        exact_matches = batch.exact_match  # [batch_size, ctx_seq_len]
        part_of_speeches = batch.part_of_speech  # [batch_size, ctx_seq_len]
        named_entity_types = batch.named_entity_type  # [batch_size, ctx_seq_len]
        normalized_term_frequencies = batch.normalized_term_frequency  # [batch_size, ctx_seq_len]

        context_mask = self.make_sequence_mask(input_sequence=context_sequence)  # [batch_size, ctx_seq_len]
        question_mask = self.make_sequence_mask(input_sequence=question_sequence)  # [batch_size, qst_seq_len]

        context_embedded = self.embedding_layer(context_sequence)  # [batch_size, ctx_seq_len, embedding_size]
        question_embedded = self.embedding_layer(question_sequence)  # [batch_size, qst_seq_len, embedding_size]

        context_aligned = self.aligned_question_embedding_layer(
            context_sequence=context_embedded,
            question_sequence=question_embedded,
            question_mask=question_mask
        )  # [batch_size, ctx_len, embedding_size]

        context_inputs = torch.cat([
            context_aligned, context_embedded, exact_matches.unsqueeze(-1), part_of_speeches.unsqueeze(-1),
            named_entity_types.unsqueeze(-1), normalized_term_frequencies.unsqueeze(-1)
        ], dim=-1)  # [batch_size, ctx_seq_len, embedding_size + hidden_size + 4]

        context_encoded = self.context_lstm_layer(
            embedded_inputs=context_inputs,
            sequence_lengths=context_lengths
        )  # [batch_size, ctx_seq_len, n_layers * hidden_size * 2]
        question_encoded = self.question_encoding_layer(
            question_embedded=question_embedded,
            question_lengths=question_lengths,
            question_mask=question_mask
        )  # [batch_size, n_layers * hidden_size * 2]

        start_scores = self.bilinear_attention_start_layer(
            context_encoded=context_encoded,
            question_encoded=question_encoded,
            context_mask=context_mask
        )  # [batch_size, ctx_seq_len]
        end_scores = self.bilinear_attention_end_layer(
            context_encoded=context_encoded,
            question_encoded=question_encoded,
            context_mask=context_mask
        )  # [batch_size, ctx_seq_len]
        return start_scores, end_scores

