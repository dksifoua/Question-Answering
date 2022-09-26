import unittest

import torch

from qa.drqa.model import DrQA


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.vocabulary_size = 500
        self.embedding_size = 300
        self.n_extra_features = 4
        self.hidden_size = 128
        self.n_layers = 5
        self.dropout = 0.5
        self.padding_index = 0

        self.drqa_model = DrQA(
            vocabulary_size=self.vocabulary_size,
            embedding_size=self.embedding_size,
            n_extra_features=self.n_extra_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            padding_index=self.padding_index
        )

    def test_forward(self):
        batch_size, ctx_seq_len, qst_seq_len = 64, 50, 20
        start_scores, end_scores = self.drqa_model(
            context_sequence=torch.randint(low=1, high=ctx_seq_len + 1, size=(batch_size, ctx_seq_len)),
            context_lengths=torch.randint(low=1, high=ctx_seq_len + 1, size=(batch_size,)),
            question_sequence=torch.randint(low=1, high=qst_seq_len + 1, size=(batch_size, qst_seq_len)),
            question_lengths=torch.randint(low=1, high=qst_seq_len + 1, size=(batch_size,)),
            exact_matches=torch.randint(low=1, high=ctx_seq_len + 1, size=(batch_size, ctx_seq_len)),
            part_of_speeches=torch.randint(low=1, high=ctx_seq_len + 1, size=(batch_size, ctx_seq_len)),
            named_entity_types=torch.randint(low=1, high=ctx_seq_len + 1, size=(batch_size, ctx_seq_len)),
            normalized_term_frequencies=torch.randn(size=(batch_size, ctx_seq_len))
        )
        self.assertEqual(start_scores.size(), torch.Size([batch_size, ctx_seq_len]))
        self.assertEqual(end_scores.size(), torch.Size([batch_size, ctx_seq_len]))

        start_indexes, end_indexes, predicted_probabilities = self.drqa_model.decode(
            starts=start_scores,
            ends=end_scores
        )
        self.assertEqual(len(start_indexes), batch_size)
        self.assertEqual(len(end_indexes), batch_size)
        self.assertEqual(len(predicted_probabilities), batch_size)
