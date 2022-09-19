import argparse
import itertools

from question_answering.io import IO
from question_answering.domain import *
from question_answering.vocabulary import Vocabulary
from question_answering.configuration import Configuration


if __name__ == "__main__":
    configuration = Configuration()
    parser = argparse.ArgumentParser(description="DrQA build vocabulary")
    parser.add_argument("--train_processed_data_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.data.processed.drqa.train_path,
                        help="Train processed data path")
    parser.add_argument("--valid_processed_data_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.data.processed.drqa.valid_path,
                        help="Validation processed data path")
    parser.add_argument("--padding_token",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.vocabulary.padding_token,
                        help="Padding token for word vocabulary")
    parser.add_argument("--unknown_token",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.vocabulary.unknown_token,
                        help="Unknown token for word vocabulary")
    parser.add_argument("--min_frequency",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.vocabulary.min_frequency,
                        help="Min frequency for each word in the vocabulary")
    parser.add_argument("--uncompleted_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.vocabulary.drqa.path,
                        help="Uncompleted path for saved vocabulary (must be completed with .format string method for "
                             "subsequent vocabularies)")
    args = parser.parse_args()

    print("Loading processed data...")
    train_qas = IO.load_from_pickle(path=args.train_processed_data_path)
    valid_qas = IO.load_from_pickle(path=args.valid_processed_data_path)
    print("Loading processed data... ok")
    print(f"Length of train qa pairs: {len(train_qas):,}")
    print(f"Length of valid qa pairs: {len(valid_qas):,}")

    print("Building vocabularies...")
    contexts, questions = zip(*map(lambda qa: (qa.context, qa.question), train_qas))
    part_of_speeches = [*itertools.chain.from_iterable(map(lambda qa: qa.token_feature.part_of_speech, train_qas))]
    named_entity_types = [*itertools.chain.from_iterable(map(lambda qa: qa.token_feature.named_entity_type, train_qas))]

    text_vocabulary = Vocabulary(padding_token=args.padding_token, unknown_token=args.unknown_token)
    part_of_speech_vocabulary = Vocabulary(padding_token=args.padding_token, unknown_token=args.unknown_token)
    named_entity_types_vocabulary = Vocabulary(padding_token=args.padding_token, unknown_token=args.unknown_token)

    text_vocabulary.build(data=contexts + questions, min_word_frequency=args.min_frequency)
    part_of_speech_vocabulary.build(data=part_of_speeches, min_word_frequency=0)
    named_entity_types_vocabulary.build(data=named_entity_types, min_word_frequency=0)
    print("Building vocabularies... ok")
    print(f"Length of text vocabulary: {len(text_vocabulary):,}")
    print(f"Length of part of speech vocabulary: {len(part_of_speech_vocabulary):,}")
    print(f"Length of named entity type vocabulary: {len(named_entity_types_vocabulary):,}")

    print("Saving vocabularies...")
    text_vocabulary.save(path=args.uncompleted_path.format("text"))
    part_of_speech_vocabulary.save(path=args.uncompleted_path.format("part_of_speech"))
    named_entity_types_vocabulary.save(path=args.uncompleted_path.format("named_entity_types"))
    print("Saving vocabularies... ok")
