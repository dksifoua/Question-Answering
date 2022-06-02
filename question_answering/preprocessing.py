import tqdm
import spacy
from typing import Dict, List

from question_answering.domain import SquadV1DataItem, Target


def parse_squad_v1_data(data: Dict, spacy_nlp: spacy.language.Language) -> List[SquadV1DataItem]:
    qas = []
    for paragraphs in tqdm.tqdm(data["data"]):
        for paragraph in paragraphs["paragraphs"]:
            context = spacy_nlp(paragraph["context"], disable=["parser", "lemmatizer"])
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = spacy_nlp(qa["question"], disable=["ner", "parser", "tagger", "lemmatizer"])
                for answer in qa["answers"]:
                    qas.append(
                        SquadV1DataItem(id_=id_, context=context, question=question,
                                        answer=spacy_nlp(answer["text"],
                                                         disable=["parser", "tagger", "ner", "lemmatizer"]),
                                        answer_start_index=answer["answer_start"]))
    return qas


def test_answer_start_indexes(qas: List[SquadV1DataItem]) -> None:
    for qa in tqdm.tqdm(qas):
        assert qa.answer.text == qa.context.text[qa.answer_start_index:qa.answer_start_index + len(qa.answer.text)]


def add_targets_to_squad_v1_data(qas: List[SquadV1DataItem]) -> None:
    for qa in tqdm.tqdm(qas):
        for i in range(len(qa.context)):
            if qa.context[i].idx == qa.answer_start_index:
                answer = qa.context[i:i + len(qa.answer)]
                qa.target = Target(start_index=answer[0].i, end_index=answer[-1].i)


def is_bad_item(qa: SquadV1DataItem) -> bool:
    """Return True if either the target is None or target indexes don't match the answer"""
    if qa.target is None:
        return False
    return qa.answer.text == qa.context[qa.target.start_index:qa.target.end_index + 1].text


def test_targets(qas: List[SquadV1DataItem]) -> None:
    for qa in qas:
        assert qa.answer.text == qa.context[qa.target.start_index:qa.target.end_index + 1].text
