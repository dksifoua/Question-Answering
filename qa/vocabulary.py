import collections
from typing import Dict, List, Union, Tuple

from spacy.tokens import Doc

from .io import IO


class Vocabulary:

    def __init__(self, padding_token: str, unknown_token: str):
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        self.vocabulary: List[str] = []
        self.word2count: Dict[str, int] = {}
        self.word2index: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {}

    def build(self, data: List[Union[Doc, str]], min_word_frequency: int) -> None:
        words = []

        type_ = type(data[0])
        if type_ == Doc:
            for item in data:  # Doc
                words += [word.text.lower() for word in item]
        elif type_ == str:
            words += data
        else:
            raise TypeError(f"The type {type_} is not supported!")

        self.word2count = collections.Counter(words)
        self.vocabulary = [self.padding_token] + sorted(filter(
            lambda word: self.word2count[word] >= min_word_frequency,
            self.word2count
        )) + [self.unknown_token]  # Ensure that pad token gets index 0 and unknown token gets the las index
        self.word2index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index2word = {index: word for index, word in enumerate(self.vocabulary)}

    def __len__(self) -> int:
        return len(self.vocabulary)

    def stoi(self, word: str) -> int:
        """
        Return the index of the word in the vocabulary.
        Return the index of the unknown token if that word doesn't exist in the vocabulary.
        """
        return self.word2index.get(word, self.word2index[self.unknown_token])

    def itos(self, index: int) -> str:
        """Return the word of the index in the vocabulary."""
        return self.index2word[index]

    def save(self, path: str) -> None:
        IO.save_to_pickle(data=self, path=path)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        return IO.load_from_pickle(path=path, return_type=Vocabulary)


class CharacterVocabulary:

    def __init__(self, padding_token: str, unknown_token: str):
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        self.vocabulary: List[str] = []
        self.character2count: Dict[str, int] = {}
        self.character2index: Dict[str, int] = {}
        self.index2character: Dict[int, str] = {}

    def build(self, data: List[Union[Doc, str, Tuple]], min_character_frequency: int) -> None:
        characters = []

        type_0 = type(data[0])
        if type_0 == Doc:
            for item in data:  # context and question
                for word in item:
                    characters += [*word.text.lower().strip()]
        else:
            raise Exception
        self.character2count = collections.Counter(characters)
        self.vocabulary = [self.padding_token] + sorted(filter(
            lambda character: self.character2count[character] >= min_character_frequency,
            self.character2count
        )) + [self.unknown_token]  # Ensure that pad token gets index 0 and unknown token gets the last index
        self.character2index = {character: index for index, character in enumerate(self.vocabulary)}
        self.index2character = {index: character for index, character in enumerate(self.vocabulary)}

    def __len__(self):
        return len(self.vocabulary)

    def ctoi(self, character: str):
        return self.character2index.get(str(character), self.character2index[self.unknown_token])

    def itoc(self, index):
        return self.index2character[index]
