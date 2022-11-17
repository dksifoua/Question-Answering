import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayNetworkLayer(nn.Module):
    """
    Highway network is a new architecture designed to ease gradient-based training of very deep networks since they
    allow unimpeded information flow across several layers on *information highways*. The architecture is characterized
    by the use of gating units which learn to regulate the flow of information through a network. Highway networks with
    hundreds of layers can be trained directly using stochastic gradient descent and with a variety of activation
    functions, opening up the possibility of studying extremely deep and efficient architectures.
    """

    def __init__(self, hidden_size: int, n_layers: int):
        super(HighwayNetworkLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dense_flow = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.dense_gate = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        @param inputs: FloatTensor[batch_size, seq_len, hidden_size]
        @return: FloatTensor[batch_size, seq_len, hidden_size]
        """
        for i in range(self.n_layers):
            flow = F.relu(self.dense_flow[i](inputs))
            gate = torch.sigmoid(self.dense_gate[i](inputs))
            inputs = gate * flow + (1 - gate) * inputs

        return inputs
