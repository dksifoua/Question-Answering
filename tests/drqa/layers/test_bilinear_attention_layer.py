import unittest
import numpy as np

import torch

from qa.drqa.layers import BiLinearAttentionLayer


class TestBiLinearAttentionLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.context_hidden_size = 128
        self.question_hidden_size = 128

        self.bilinear_attention_layer = BiLinearAttentionLayer(
            context_hidden_size=self.context_hidden_size,
            question_hidden_size=self.question_hidden_size
        )

    def test_forward(self):
        batch_size, ctx_seq_len, qst_seq_len = 64, 50, 20
        outputs = self.bilinear_attention_layer(
            context_encoded=torch.randn(size=(batch_size, ctx_seq_len, self.context_hidden_size)),
            question_encoded=torch.randn(size=(batch_size, self.question_hidden_size)),
            context_mask=torch.Tensor(np.random.randint(low=0, high=2, size=(batch_size, ctx_seq_len)))
        )
        self.assertEqual(
            outputs.size(), torch.Size([batch_size, ctx_seq_len])
        )
