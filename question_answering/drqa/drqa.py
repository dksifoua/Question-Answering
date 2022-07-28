from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StackedBiLSTMsLayer(nn.Module):

    def __init__(self, embedding_size: int, hidden_size: int, n_layers: int, dropout: float):
        super(StackedBiLSTMsLayer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_layers = nn.ModuleList([nn.LSTM(embedding_size if i == 0 else hidden_size * 2, hidden_size,
                                                  batch_first=True, num_layers=n_layers, bidirectional=True)
                                          for i in range(n_layers)])

    def apply_lstm(self, layer: nn.LSTM, inputs: torch.Tensor, lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param layer:
        :param inputs: Float[batch_size, seq_len, embedding_size | hidden_size * 2]
        :param lengths: Long[batch_size, seq_len]
        :return [batch_size, seq_len, hidden_size * 2]
        """
        inputs = self.dropout(inputs)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = layer(packed)
        out_padded, out_lengths = pad_packed_sequence(out_packed, batch_first=True)
        # [batch_size, seq_len, hidden_size * 2]
        return out_padded, out_lengths
