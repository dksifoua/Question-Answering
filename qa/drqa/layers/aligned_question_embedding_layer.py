import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AlignedQuestionEmbeddingLayer(nn.Module):

    def __init__(self, embedding_size: int, hidden_size: int):
        """
        Aligned Question Embedding
        $$f_{align}(p_i) = \sum_j a_{i, j}E(q_j)$$
        where the attention score $a_{i, j}$ captures the similarity between $p_i$ and each question words $q_j$.
        Specifically, $a_{i, j}$ is computed by the dot products between nonlinear mappings of word embeddings:
        $$a_{i, j} = \frac{exp(\alpha(E(p_i)).\alpha(E(q_j)))}{\sum_{j'}exp(\alpha(E(p_i)).\alpha(E(q_{j'})))}$$,
        and $\alpha(.)$ is a single dense layer with ReLU non-linearity. Compared to the exact match features, these
        features add soft alignments between similar but non-identical words (e.g., car and vehicle)

        :param embedding_size: Size of word embedding.
        :param hidden_size: Hidden size of the dense layer.
        """
        super(AlignedQuestionEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.dense = nn.Linear(in_features=embedding_size, out_features=hidden_size)

    def forward(self, context_sequence: Tensor, question_sequence: Tensor, question_mask: Tensor) -> Tensor:
        """
        :param context_sequence: Sequence context inputs. FloatTensor[batch_size, ctx_seq_len, embedding_size]
        :param question_sequence: Sequence question inputs. FloatTensor[batch_size, qst_seq_len, embedding_size]
        :param question_mask: Mask of question regarding if it is a padding token or not.
            IntTensor[batch_size, qst_seq_len]
        :return: FloatTensor[batch_size, ctx_seq_len, hidden_size]
        """
        context_logits = F.relu(self.dense(context_sequence))  # [batch_size, ctx_seq_len, hidden_size]
        question_logits = F.relu(self.dense(question_sequence))  # [batch_size, qst_seq_len, hidden_size]
        scores = torch.bmm(context_logits, question_logits.transpose(-1, -2))  # [batch_size, ctx_seq_len, qst_seq_len]
        # Mask scores in order to force attention weights corresponding to padding tokens to be 0.
        scores = scores.masked_fill(question_mask.unsqueeze(1) == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, ctx_seq_len, qst_seq_len]
        return torch.bmm(attention_weights, question_logits)
