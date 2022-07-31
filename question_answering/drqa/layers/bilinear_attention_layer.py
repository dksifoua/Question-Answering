import torch
import torch.nn as nn
from torch import Tensor


class BiLinearAttentionLayer(nn.Module):
    
    def __init__(self, context_hidden_size: int, question_hidden_size: int):
        """
        At the paragraph level, the goal is to predict the span of tokens that is most likely the correct answer.
        We take the paragraph vectors $\{p_1, ... , p_m\}$ and the question vector $q$ as input, and simply train two
        classifiers independently for predicting the two ends of the span.
        Concretely, we use a bi-linear term to capture the similarity between $p_i$ and $q$ and compute the
        probabilities of each token being start and end as:
        $$
        P_{start}(i) \propto exp(p_i W_s q) \\
        P_{end}(i) \propto exp(p_i W_e q)
        $$

        :param context_hidden_size:
        :param question_hidden_size:
        """
        super(BiLinearAttentionLayer, self).__init__()
        self.context_hidden_size = context_hidden_size
        self.question_hidden_size = question_hidden_size

        self.dense = nn.Linear(in_features=question_hidden_size, out_features=context_hidden_size, bias=False)

    def forward(self, context_encoded: Tensor, question_encoded: Tensor, context_mask: Tensor) -> Tensor:
        """
        :param context_encoded: FloatTensor[batch_size, ctx_seq_len, ctx_hid_size]
        :param question_encoded: FloatTensor[batch_size, qst_seq_len]
        :param context_mask: IntTensor[batch_size, ctx_seq_len]
        :return FloatTensor[batch_size, ctx_seq_len]
        """
        question_encoded = self.dense(question_encoded)  # [batch_size, ctx_hid_size]
        scores = torch.bmm(context_encoded, question_encoded.unsqueeze(-1)).squeeze(-1)  # [batch_size, ctx_seq_len]
        return scores.masked_fill(context_mask == 0, float("-inf"))

