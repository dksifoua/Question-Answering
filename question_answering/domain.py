import dataclasses
from typing import NamedTuple

from spacy.tokens import Doc

Target = NamedTuple("Target", [("start_index", int), ("end_index", int)])


@dataclasses.dataclass
class SquadV1DataItem:
    id_: str
    context: Doc
    question: Doc
    answer: Doc
    answer_start_index: int
    target: Target = None
