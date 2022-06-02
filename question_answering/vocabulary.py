from typing import List, Union


class SquadV1Vocabulary:

    def __init__(self, padding_token: str, unknown_token: str):
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        self.vocabulary = None
        self.word2count = None
        self.word2index = None
        self.index2word = None

    def build(self, data: List[Union], min_freq: int) -> None:
        words = [self.padding_token, self.unknown_token]
        # TODO

