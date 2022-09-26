import unittest
import numpy as np

import torch
from qa.drqa.layers import QuestionEncodingLayer


class TestQuestionEncodingLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 300
        self.hidden_size = 128
        self.n_layers = 3
        self.dropout = 0.5

        self.question_encoding_layer = QuestionEncodingLayer(
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

    def test_forward(self):
        batch_size, qst_seq_len = 128, 20
        outputs = self.question_encoding_layer(
            question_embedded=torch.randn(size=(batch_size, qst_seq_len, self.embedding_size)),
            question_lengths=torch.randint(low=1, high=qst_seq_len + 1, size=(batch_size,)),
            question_mask=torch.Tensor(np.random.randint(low=0, high=2, size=(batch_size, qst_seq_len)))
        )
        self.assertEqual(
            outputs.size(), torch.Size([batch_size, self.n_layers * self.hidden_size * 2])
        )
