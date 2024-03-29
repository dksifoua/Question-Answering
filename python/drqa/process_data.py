import spacy
import random
import argparse

from qa.io import IO
from qa.configuration import Configuration
from qa.logger import QALogger
from qa.preprocessing import add_extra_features_squad_v1, add_targets_to_squad_v1_data, is_bad_item, \
    parse_squad_v1_data, test_answer_start_indexes, test_targets


if __name__ == "__main__":
    logger = QALogger.get_logger(name="ProcessData")
    configuration = Configuration()
    parser = argparse.ArgumentParser(description="DrQA data processing")
    parser.add_argument("--train_raw_data_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.data.raw.train_path,
                        help="Train raw data path")
    parser.add_argument("--valid_raw_data_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.data.raw.valid_path,
                        help="Validation raw data path")
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
    args = parser.parse_args()

    language = spacy.load(name="en_core_web_lg")

    logger.info("Loading data...")
    train_raw_data = IO.load_from_json(path=args.train_raw_data_path)
    valid_raw_data = IO.load_from_json(path=args.valid_raw_data_path)
    logger.info("Loading data... ok")
    logger.info("Length of raw train data: {:,}".format(len(train_raw_data["data"])))
    logger.info("Length of raw valid data: {:,}".format(len(valid_raw_data["data"])))

    logger.info("parsing JSON data...")
    train_qas = parse_squad_v1_data(data=train_raw_data, spacy_nlp=language)
    valid_qas = parse_squad_v1_data(data=valid_raw_data, spacy_nlp=language)
    logger.info("parsing JSON data... ok")
    logger.info(f"Length of train qa pairs: {len(train_qas):,}")
    logger.info(f"Length of valid qa pairs: {len(valid_qas):,}")
    logger.info(f"Train example: {train_qas[random.randint(a=0, b=len(train_qas) - 1)]}")

    logger.info("Testing start indexes...")
    test_answer_start_indexes(qas=train_qas)
    test_answer_start_indexes(qas=valid_qas)
    logger.info("Testing start indexes... ok")

    logger.info("Adding targets...")
    add_targets_to_squad_v1_data(qas=train_qas)
    add_targets_to_squad_v1_data(qas=valid_qas)
    logger.info("Adding targets... ok")

    logger.info("Filtering out bad targets...")
    train_qas = [*filter(is_bad_item, train_qas)]
    valid_qas = [*filter(is_bad_item, valid_qas)]
    logger.info("Filtering out bad targets... ok")
    logger.info(f"Length of train qa pairs after filtering out bad qa pairs: {len(train_qas):,}")
    logger.info(f"Length of valid qa pairs after filtering out bad qa pairs: {len(valid_qas):,}")

    logger.info("Testing targets...")
    test_targets(qas=train_qas)
    test_targets(qas=valid_qas)
    logger.info("Testing targets... ok")

    logger.info("Adding extra features...")
    train_qas = add_extra_features_squad_v1(qas=train_qas)
    valid_qas = add_extra_features_squad_v1(qas=valid_qas)
    logger.info("Adding extra features... ok")

    logger.info("Saving processed data...")
    IO.save_to_pickle(data=train_qas, path=args.train_processed_data_path)
    IO.save_to_pickle(data=valid_qas, path=args.valid_processed_data_path)
    logger.info("Saving processed data... ok")
