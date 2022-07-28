import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple


class StackedBiLSTMsLayer(nn.Module):

    def __init__(self, embedding_size: int, hidden_size: int, n_layers: int, dropout: float):
        super(StackedBiLSTMsLayer).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=embedding_size if i == 0 else hidden_size * 2, hidden_size=hidden_size,
                batch_first=True, num_layers=n_layers, bidirectional=True
            ) for i in range(n_layers)
        ])

    def __forward_lstm(self, layer: nn.Module, inputs: torch.Tensor, lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.dropout(inputs)
        packed_inputs = pack_padded_sequence(input=inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = layer(packed_inputs)
        padded_outputs, outputs_lengths = pad_packed_sequence(sequence=packed_outputs, batch_first=True)
        return padded_outputs, outputs_lengths

    def forward(self, input_embedded: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        outputs, lens = [input_embedded], sequence_lengths
        for lstm_layer in self.lstm_layers:
            out, lens = self.apply_lstm(layer=lstm_layer, inputs=outputs[-1], lengths=lens)
            outputs.append(out)
        return self.dropout(torch.cat(outputs[1:], dim=-1))