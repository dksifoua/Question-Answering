import unittest

import torch

from bidaf.layers.highway_network_layer import HighwayNetworkLayer


class TestHighwayNetwork(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_size = 128
        self.n_layers = 5
        self.highway_network_layer = HighwayNetworkLayer(hidden_size=self.hidden_size, n_layers=self.n_layers)

    def test_forward(self):
        batch_size, input_seq_len = 64, 50
        outputs = self.highway_network_layer(inputs=torch.randn(size=(batch_size, input_seq_len, self.hidden_size)))
        self.assertEqual(outputs.size(), torch.Size([batch_size, input_seq_len, self.hidden_size]))
