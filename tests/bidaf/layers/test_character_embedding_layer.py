import unittest

import torch

from qa.bidaf.layers.character_embedding_layer import CharacterEmbeddingLayer


class TestCharacterEmbeddingLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.vocabulary_size = 500
        self.character_embedding_size = 300
        self.token_embedding_size = 300
        self.kernel_size = 3
        self.padding_index = 0
        self.character_embedding_layer = CharacterEmbeddingLayer(
            vocabulary_size=self.vocabulary_size,
            character_embedding_size=self.character_embedding_size,
            token_embedding_size=self.token_embedding_size,
            kernel_size=self.kernel_size,
            padding_index=self.padding_index
        )

    def test_forward(self) -> None:
        batch_size, input_seq_len, input_char_len = 64, 60, 120
        outputs = self.character_embedding_layer(
            character_sequence_inputs=torch.randint(
                low=1,
                high=input_char_len + 1,
                size=(batch_size, input_seq_len, input_char_len)
            )
        )
        print(outputs.size())
        self.assertEqual(
            outputs.size(), torch.Size([batch_size, input_seq_len, self.token_embedding_size])
        )
