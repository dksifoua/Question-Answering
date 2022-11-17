import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEmbeddingLayer(nn.Module):

    def __init__(self, vocabulary_size: int, character_embedding_size: int, token_embedding_size: int,
                 kernel_size: int, padding_index: int):
        """
        Character embedding layer is responsible for mapping each word to a high-dimensional vector space.
        Let ${x_1, ..., x_T}$ and ${q_1, ..., q_J}$ represent the words in the input context paragraph and query,
        respectively. Following Kim (2014), we obtain the character level embedding of each word using Convolutional
        Neural Networks (CNN). Characters are embedded into vectors, which can be considered as 1D inputs to the CNN,
        and whose size is the input channel size of the CNN. The outputs of the CNN are max-pooled over the entire
        width to obtain a fixed-size vector for each word.

        @param vocabulary_size:
        @param character_embedding_size:
        @param token_embedding_size:
        @param kernel_size:
        @param padding_index:
        """
        super(CharacterEmbeddingLayer, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.character_embedding_size = character_embedding_size
        self.token_embedding_size = token_embedding_size
        self.kernel_size = kernel_size
        self.padding_index = padding_index
        self.embedding = nn.Embedding(vocabulary_size, character_embedding_size, padding_idx=padding_index)
        self.cond2d = nn.Conv2d(1, token_embedding_size, kernel_size=(character_embedding_size, kernel_size))

    def forward(self, character_sequence_inputs: torch.Tensor) -> torch.Tensor:
        """

        @param character_sequence_inputs: LongTensor[batch_size, seq_len, char_len]
        @return: FloatTensor[batch_size, seq_len, token_embedding_size]
        """
        embedded = self.embedding(character_sequence_inputs)  # [batch_size, seq_len, char_len, char_embedding_size]
        embedded = embedded.transpose(-1, -2)  # [batch_size, seq_len, char_embedding_size, char_len]
        embedded = embedded.view(-1, self.character_embedding_size,
                                 embedded.size(-1))  # [batch_size * seq_len, char_embedding_size, char_len]
        embedded = embedded.unsqueeze(1)  # [batch_size * seq_len, 1, char_embedding_size, char_len]
        convoluted = F.relu(
            self.cond2d(embedded))  # [batch_size * seq_len, token_embedding_size, 1, char_len - kernel_size + 1]
        convoluted = convoluted.squeeze(2)  # [batch_size * seq_len, token_embedding_size, char_len - kernel_size + 1]
        outputs = F.max_pool1d(convoluted, kernel_size=convoluted.size(-1))
        # [batch_size * seq_len, token_embedding_size, 1]
        # print(out.shape, '[batch_size * seq_len, token_embedding_size, 1, char_len - kernel_size + 1]')
        outputs = outputs.squeeze(-1)  # [batch_size * seq_len, token_embedding_size]
        outputs = outputs.view(character_sequence_inputs.size(0), -1, outputs.size(-1))
        # [batch_size, seq_len, token_embedding_size]
        # May be applied bias + tanh non-linearity???
        return outputs
