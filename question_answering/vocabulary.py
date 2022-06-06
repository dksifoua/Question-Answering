import collections
from typing import List, Union

from spacy.tokens import Doc


class SquadV1Vocabulary:

    def __init__(self, padding_token: str, unknown_token: str):
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        self.vocabulary = None
        self.word2count = None
        self.word2index = None
        self.index2word = None

    def build(self, data: List[Union[Doc, str]], min_word_frequency: int) -> None:
        words = [self.padding_token, self.unknown_token]

        type_ = type(data[0])
        if type_ == Doc:
            for item in data:  # Doc
                words += [word.text.lower() for word in item]
        elif type_ == str:
            words += data
        else:
            raise TypeError(f"The type {type_} is not supported!")

        self.word2count = collections.Counter(words)
        self.vocabulary = sorted(filter(
                lambda word:
                self.word2count[word] >= min_word_frequency
                or word == self.padding_token
                or word == self.unknown_token, self.word2count
        ))
        self.word2index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index2word = {index: word for index, word in enumerate(self.vocabulary)}

    def __len__(self) -> int:
        return len(self.vocabulary)

    def stoi(self, word: str) -> int:
        """
        Return the index of the word in the vocabulary.
        Return the index of the unknown token if that word doesn't exist in the vocabulary.
        """
        return self.word2index.get(str(word), self.word2index[self.unknown_token])

    def itos(self, index: int) -> str:
        """Return the word of the index in the vocabulary."""
        return self.index2word[index]
