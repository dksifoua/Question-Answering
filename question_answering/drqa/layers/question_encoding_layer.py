import torch
import torch.nn as nn
import torch.nn.functional as F
from question_answering.drqa.layers import StackedBiLSTMsLayer


class QuestionEncodingLayer(nn.Module):
    
    def __init__(self, embedding_size: int, hidden_size: int, n_layers: int, dropout: float):
        """
        The question encoding consists of applying a recurrent neural network on top of the word embeddings of $q_i$
        and combine the resulting hidden units into one single vector: $\{q_1, ..., q_l\} \rightarrow q$. We compute
        $q = \sum_j{b_j q_j}$ where $b_j$ encodes the importance of each question word:
        $$b_j = \frac{exp(w.q_j)}{\sum_{j'}{exp(w.q_{j'})}}$$
        and $w$ is a weight vector to learn

        :param embedding_size:
        :param hidden_size:
        :param n_layers:
        :param dropout:
        """
        super(QuestionEncodingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        self.stacked_bilstm_layers = StackedBiLSTMsLayer(
            embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout
        )
        self.dense = nn.Linear(in_features=embedding_size, out_features=1, bias=False)

    def __linear_self_attention(self, question_embedded: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        """
        :param question_embedded: FloatTensor[batch_size, qst_seq_len, embedding_size]
        :param question_mask: Mask of question regarding if it is a padding token or not.
            IntTensor[batch_size, qst_seq_len]
        :return: FloatTensor[batch_size, qst_seq_len]
        """
        scores = self.dense(question_embedded).squeeze(-1)  # [batch_size, qst_seq_len]
        scores = scores.masked_fill(question_mask == 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    def forward(self, question_embedded: torch.Tensor, question_lengths: torch.Tensor, question_mask: torch.Tensor) \
            -> torch.Tensor:
        """
        :param question_embedded: FloatTensor[batch_size, qst_seq_len, embedding_size]
        :param question_lengths: Sequence question lengths. LongTensor[batch_size,]
        :param question_mask: Mask of question regarding if it is a padding token or not.
            IntTensor[batch_size, qst_seq_len]
        :return: FloatTensor[batch_size, n_layers * hidden_size * 2]
        """
        lstm_outputs = self.stacked_bilstm_layers(embedded_inputs=question_embedded, sequence_lengths=question_lengths)
        # lstm_outputs: [batch_size, qst_seq_len, n_layers * hidden_size * 2]
        attention_weights = self.__linear_self_attention(
            question_embedded=question_embedded, question_mask=question_mask
        )  # [batch_size, qst_seq_len]
        return torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
