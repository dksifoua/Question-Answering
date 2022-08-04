import torch
import unittest
from question_answering.drqa.layers import StackedBiLSTMsLayer


class TestStackedBiLSTMsLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 300
        self.hidden_size = 64
        self.n_layers = 8
        self.dropout = 0.5

        self.stack_bilstms_layer = StackedBiLSTMsLayer(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                                       n_layers=self.n_layers, dropout=self.dropout)

    def test_forward(self):
        sequence_length, batch_size = 50, 32
        bilstm_outputs = self.stack_bilstms_layer(
            embedded_inputs=torch.randn(size=(batch_size, sequence_length, self.embedding_size)),
            sequence_lengths=torch.randint(low=10, high=sequence_length, size=(batch_size,))
        )
        self.assertEqual(
            bilstm_outputs.size(), torch.Size([batch_size, sequence_length, self.n_layers * self.hidden_size * 2])
        )
