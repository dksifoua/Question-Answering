import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple


class StackedBiLSTMsLayer(nn.Module):

    def __init__(self, embedding_size: int, hidden_size: int, n_layers: int, dropout: float):
        """
        :param embedding_size: Size of word embedding.
        :param hidden_size: Hidden size of lstm layers.
        :param n_layers: Number of layers.
        :param dropout: Dropout value in [0, 1).
        """
        super(StackedBiLSTMsLayer, self).__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("Dropout must be in [0, 1)")

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

    def __forward_lstm(self, layer: nn.Module, inputs: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param layer: LSTM layer.
        :param inputs: Sequence inputs. FloatTensor[batch_size, max_sequence_length, embedding_size|hidden_size * 2]
        :param lengths: Sequence lengths. LongTensor[batch_size,]
        :return:
            padded_outputs FloatTensor[batch_size, max_sequence_length, hidden_size * 2]
            outputs_lengths LongTensor[batch_size,]
        """
        inputs = self.dropout(inputs)
        packed_inputs = pack_padded_sequence(input=inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = layer(packed_inputs)
        padded_outputs, outputs_lengths = pad_packed_sequence(sequence=packed_outputs, batch_first=True)
        return padded_outputs, outputs_lengths

    def forward(self, embedded_inputs: Tensor, sequence_lengths: Tensor) -> Tensor:
        """
        :param embedded_inputs: Sequence inputs. FloatTensor[batch_size, max_sequence_length, embedding_size]
        :param sequence_lengths: Sequence lengths. LongTensor[batch_size,]
        :return: FloatTensor[batch_size, sequence_lengths, n_layers * hidden_size * 2]
        """
        outputs, lens = [embedded_inputs], sequence_lengths
        for lstm_layer in self.lstm_layers:
            out, lens = self.__forward_lstm(layer=lstm_layer, inputs=outputs[-1], lengths=lens)
            outputs.append(out)
        return self.dropout(torch.cat(outputs[1:], dim=-1))
