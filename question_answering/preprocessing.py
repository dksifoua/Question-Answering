import tqdm
import spacy
import collections
from spacy.tokens import Token
from typing import Dict, List

from question_answering.domain import DrQARawDatasetItem, RawDatasetItem, Target, TokenFeature


def parse_squad_v1_data(data: Dict, spacy_nlp: spacy.language.Language) -> List[RawDatasetItem]:
    qas = []
    disabled_components = ["parser", "lemmatizer", "tagger", "ner"]
    for paragraphs in tqdm.tqdm(data["data"]):
        for paragraph in paragraphs["paragraphs"]:
            context = spacy_nlp(paragraph["context"], disable=disabled_components[:1])
            for qa in paragraph["qas"]:
                id_ = qa["id"]
                question = spacy_nlp(qa["question"], disable=disabled_components)
                for answer in qa["answers"]:
                    qas.append(RawDatasetItem(id_=id_, context=context, question=question,
                                              answer=spacy_nlp(answer["text"], disable=disabled_components),
                                              answer_start_index=answer["answer_start"]))
    return qas


def test_answer_start_indexes(qas: List[RawDatasetItem]) -> None:
    for qa in tqdm.tqdm(qas):  # type: RawDatasetItem
        assert qa.answer.text == qa.context.text[qa.answer_start_index:qa.answer_start_index + len(qa.answer.text)]


def add_targets_to_squad_v1_data(qas: List[RawDatasetItem]) -> None:
    for qa in tqdm.tqdm(qas):  # type: RawDatasetItem
        for i in range(len(qa.context)):
            if qa.context[i].idx == qa.answer_start_index:
                answer = qa.context[i:i + len(qa.answer)]
                qa.target = Target(start_index=answer[0].i, end_index=answer[-1].i)


def is_bad_item(qa: RawDatasetItem) -> bool:
    """Return True if either the target is None or target indexes don't match the answer. Return False otherwise"""
    if qa.target is None:
        return False
    return qa.answer.text == qa.context[qa.target.start_index:qa.target.end_index + 1].text


def test_targets(qas: List[RawDatasetItem]) -> None:
    for qa in qas:
        assert qa.answer.text == qa.context[qa.target.start_index:qa.target.end_index + 1].text


def add_extra_features_squad_v1(qas: List[RawDatasetItem]) -> List[DrQARawDatasetItem]:
    """Add extra features: Exact Match, Part-of-Speech, Name Entity Recognition & Normalized Term Frequency"""
    qa_token_features = []
    for qa in tqdm.tqdm(qas):  # type: RawDatasetItem
        question = [token.text.lower() for token in qa.question]
        count_context_tokens = collections.Counter(map(lambda token: token.text.lower(), qa.context))

        frequency_context_tokens: Dict[int, int] = {}
        for index, token in enumerate(qa.context):  # type: int, Token
            frequency_context_tokens[index] = count_context_tokens[token.text.lower()]
        norm_frequency_context_tokens = sum(frequency_context_tokens.values())

        token_feature = TokenFeature(
            exact_match=[qa.context[index].text.lower() in question for index in range(len(qa.context))],
            part_of_speech=[qa.context[index].tag_ for index in range(len(qa.context))],
            named_entity_type=[qa.context[index].ent_type_ for index in range(len(qa.context))],
            normalized_term_frequency=[
                frequency_context_tokens[index] / norm_frequency_context_tokens for index in range(len(qa.context))
            ]
        )

        qa_token_features.append(
            DrQARawDatasetItem(id_=qa.id_, context=qa.context, question=qa.question, answer=qa.answer,
                               answer_start_index=qa.answer_start_index, token_feature=token_feature)
        )

    return qa_token_features
