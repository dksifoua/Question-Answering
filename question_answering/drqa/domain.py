import torch
import dataclasses
from typing import List, NamedTuple, Tuple, Union

from spacy.tokens import Doc


class Target(NamedTuple):
    start_index: int
    end_index: int


class TokenFeature(NamedTuple):
    exact_match: List[bool]
    part_of_speech: List[str]
    named_entity_type: List[str]
    normalized_term_frequency: List[float]


@dataclasses.dataclass
class SquadV1DataItem:
    id_: str
    context: Doc
    question: Doc
    answer: Doc
    answer_start_index: int
    target: Target = None
    token_feature: TokenFeature = None


class SquadV1DatasetItem(NamedTuple):
    context: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
    question: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
    target: torch.LongTensor
    exact_match: torch.LongTensor
    part_of_speech: torch.LongTensor
    named_entity_type: torch.LongTensor
    normalized_term_frequency: torch.FloatTensor
