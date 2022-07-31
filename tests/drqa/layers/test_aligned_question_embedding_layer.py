import torch
import unittest
import numpy as np
from question_answering.drqa.layers import AlignedQuestionEmbeddingLayer


class TestAlignedQuestionEmbeddingLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 300
        self.hidden_size = 128

        self.aligned_question_embedding_layer = AlignedQuestionEmbeddingLayer(
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size
        )

    def test_forward(self):
        batch_size, ctx_seq_len, qst_seq_len = 64, 50, 20
        aligned_question_embedding = self.aligned_question_embedding_layer(
            context_sequence=torch.randn(size=(batch_size, ctx_seq_len, self.embedding_size)),
            question_sequence=torch.randn(size=(batch_size, qst_seq_len, self.embedding_size)),
            question_mask=torch.IntTensor(np.random.randint(low=0, high=2, size=(batch_size, qst_seq_len)))
        )
        self.assertEqual(
            aligned_question_embedding.size(), torch.Size([batch_size, ctx_seq_len, self.hidden_size])
        )
