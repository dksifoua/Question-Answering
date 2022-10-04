import dataclasses
from typing import List, Tuple, Union

import torch
from spacy.tokens import Doc


@dataclasses.dataclass
class _UnpackingDataClassMixin:

    def __iter__(self):
        return iter(dataclasses.astuple(self))


@dataclasses.dataclass
class Target(_UnpackingDataClassMixin):
    start_index: int
    end_index: int


@dataclasses.dataclass
class RawDatasetItem(_UnpackingDataClassMixin):
    id_: str
    context: Doc
    question: Doc
    answer: Doc
    answer_start_index: int
    target: Target = None


@dataclasses.dataclass
class TokenFeature(_UnpackingDataClassMixin):
    exact_match: List[bool]
    part_of_speech: List[str]
    named_entity_type: List[str]
    normalized_term_frequency: List[float]


@dataclasses.dataclass
class DrQARawDatasetItem(RawDatasetItem):
    token_feature: TokenFeature = None


@dataclasses.dataclass
class DrQATensorDatasetItem(_UnpackingDataClassMixin):
    id_: torch.LongTensor
    context: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
    question: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
    target: torch.LongTensor
    exact_match: torch.LongTensor
    part_of_speech: torch.LongTensor
    named_entity_type: torch.LongTensor
    normalized_term_frequency: torch.FloatTensor


DrQATensorDatasetBatch = DrQATensorDatasetItem

__all__ = [
    "Target",
    "RawDatasetItem",
    "TokenFeature",
    "DrQARawDatasetItem",
    "DrQATensorDatasetItem",
    "DrQATensorDatasetBatch"
]
