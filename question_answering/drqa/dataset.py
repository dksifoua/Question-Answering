from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from question_answering.domain import DrQARawDatasetItem, DrQATensorDatasetItem, DrQATensorDatasetBatch
from question_answering.vocabulary import Vocabulary


class SquadV1Dataset(Dataset):

    def __init__(self, data: List[DrQARawDatasetItem], text_vocab: Vocabulary, pos_vocab: Vocabulary,
                 ner_vocab: Vocabulary):
        self.data = data
        self.pos_vocab = pos_vocab
        self.ner_vocab = ner_vocab
        self.text_vocab = text_vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> DrQATensorDatasetItem:
        item: DrQARawDatasetItem = self.data[idx]
        context = torch.LongTensor([*map(lambda token: self.text_vocab.stoi(token.text.lower()), item.context)])
        question = torch.LongTensor([*map(lambda token: self.text_vocab.stoi(token.text.lower()), item.question)])
        target = torch.LongTensor([item.target.start_index, item.target.end_index])
        exact_match = torch.LongTensor(item.token_feature.exact_match)
        part_of_speech = torch.LongTensor(
            [*map(lambda token: self.pos_vocab.stoi(token), item.token_feature.part_of_speech)])
        named_entity_type = torch.LongTensor(
            [*map(lambda token: self.ner_vocab.stoi(token), item.token_feature.named_entity_type)])
        normalized_term_frequency = torch.FloatTensor(item.token_feature.normalized_term_frequency)
        return DrQATensorDatasetItem(
            context=context,
            question=question,
            target=target,
            exact_match=exact_match,
            part_of_speech=part_of_speech,
            named_entity_type=named_entity_type,
            normalized_term_frequency=normalized_term_frequency
        )


def add_padding_and_batch_data(batch: List[DrQATensorDatasetItem], pad_token: str, text_vocab: Vocabulary,
                               pos_vocab: Vocabulary, ner_vocab: Vocabulary, include_lengths: bool,
                               device: torch.device) -> DrQATensorDatasetBatch:
    batch_context = [item.context for item in batch]
    batch_question = [item.question for item in batch]
    batch_target = [item.target for item in batch]
    batch_exact_match = [item.exact_match for item in batch]
    batch_part_of_speech = [item.part_of_speech for item in batch]
    batch_named_entity_type = [item.named_entity_type for item in batch]
    batch_normalized_term_frequency = [item.normalized_term_frequency for item in batch]

    length_context, length_question = None, None
    if include_lengths:
        length_context = torch.LongTensor([context.size(0) for context in batch_context]).to(device)
        length_question = torch.LongTensor([question.size(0) for question in batch_question]).to(device)

    batch_padded_context = pad_sequence(batch_context,
                                        batch_first=True,
                                        padding_value=text_vocab.stoi(pad_token)).to(device)
    batch_padded_question = pad_sequence(batch_question,
                                         batch_first=True,
                                         padding_value=text_vocab.stoi(pad_token)).to(device)
    batch_padded_target = pad_sequence(batch_target, batch_first=True).to(device)
    batch_padded_exact_match = pad_sequence(batch_exact_match, batch_first=True).to(device)
    batch_padded_part_of_speech = pad_sequence(batch_part_of_speech,
                                               batch_first=True,
                                               padding_value=pos_vocab.stoi(pad_token)).to(device)
    batch_padded_named_entity_type = pad_sequence(batch_named_entity_type,
                                                  batch_first=True,
                                                  padding_value=ner_vocab.stoi(pad_token)).to(device)
    batch_padded_normalized_term_frequency = pad_sequence(batch_normalized_term_frequency,
                                                          batch_first=True).to(device)
    return DrQATensorDatasetBatch(
        context=(batch_padded_context, length_context),
        question=(batch_padded_question, length_question),
        target=batch_padded_target,
        exact_match=batch_padded_exact_match,
        part_of_speech=batch_padded_part_of_speech,
        named_entity_type=batch_padded_named_entity_type,
        normalized_term_frequency=batch_padded_normalized_term_frequency
    )
