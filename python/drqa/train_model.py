import argparse
import functools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from drqa import SquadV1Dataset, add_padding_and_batch_data
from qa.io import IO
from qa.domain import *
from drqa.model import DrQA
from qa.logger import QALogger
from qa.trainer import Trainer
from qa.utils import seed_everything, ignore_warnings, load_glove_embeddings, extract_embeddings
from qa.vocabulary import Vocabulary
from qa.configuration import Configuration


if __name__ == "__main__":
    logger = QALogger.get_logger(name="BuildVocabulary")
    configuration = Configuration()
    parser = argparse.ArgumentParser(description="Train a DrQA model")
    parser.add_argument("--seed",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.seed,
                        help="Seed for reproducibility")
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
    parser.add_argument("--embedding_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.model.embeddings.path,
                        help="Embedding path")
    parser.add_argument("--n_layers",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.model.drqa.n_layers,
                        help="Number of layers of the model")
    parser.add_argument("--embedding_size",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.model.drqa.embedding_size,
                        help="Embedding size")
    parser.add_argument("--tune_embeddings",
                        action="store",
                        type=bool,
                        required=False,
                        default=configuration.base.model.embeddings.tune,
                        help="Tune embeddings?")
    parser.add_argument("--hidden_size",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.model.drqa.hidden_size,
                        help="Hidden size")
    parser.add_argument("--dropout",
                        action="store",
                        type=float,
                        required=False,
                        default=configuration.base.model.drqa.dropout,
                        help="Dropout")
    parser.add_argument("--model_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.model.drqa.path,
                        help="Model saved path")
    parser.add_argument("--learning_rate",
                        action="store",
                        type=float,
                        required=False,
                        default=configuration.base.training.drqa.learning_rate,
                        help="Number of epochs")
    parser.add_argument("--n_epochs",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.training.drqa.n_epochs,
                        help="Number of epochs")
    parser.add_argument("--batch_size",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.training.drqa.batch_size,
                        help="Number of epochs")
    parser.add_argument("--gradient_clipping",
                        action="store",
                        type=float,
                        required=False,
                        default=configuration.base.training.drqa.gradient_clipping,
                        help="Gradient clipping value")
    parser.add_argument("--vocabulary_uncompleted_path",
                        action="store",
                        type=str,
                        required=False,
                        default=configuration.base.vocabulary.drqa.path,
                        help="Uncompleted path for saved vocabulary (must be completed with .format string method for "
                             "subsequent vocabularies)")
    args = parser.parse_args()

    ignore_warnings()
    seed_everything(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading processed data...")
    train_qas = IO.load_from_pickle(path=args.train_processed_data_path, return_type=DrQARawDatasetItem)
    valid_qas = IO.load_from_pickle(path=args.valid_processed_data_path, return_type=DrQARawDatasetItem)
    logger.info("Loading processed data... done.")
    logger.info(f"Length of train qa pairs: {len(train_qas):,}")
    logger.info(f"Length of valid qa pairs: {len(valid_qas):,}")

    logger.info("Loading vocabularies...")
    id_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("id"))
    text_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("text"))
    part_of_speech_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("part_of_speech"))
    named_entity_types_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("named_entity_types"))
    logger.info("Loading vocabularies... done.")
    logger.info(f"Length of id vocabulary: {len(id_vocabulary):,}")
    logger.info(f"Length of text vocabulary: {len(text_vocabulary):,}")
    logger.info(f"Length of part of speech vocabulary: {len(part_of_speech_vocabulary):,}")
    logger.info(f"Length of named entity type vocabulary: {len(named_entity_types_vocabulary):,}")

    logger.info(f"Building datasets...")
    train_dataset = SquadV1Dataset(
        data=train_qas,
        id_vocab=id_vocabulary,
        text_vocab=text_vocabulary,
        pos_vocab=part_of_speech_vocabulary,
        ner_vocab=named_entity_types_vocabulary
    )
    valid_dataset = SquadV1Dataset(
        data=valid_qas,
        id_vocab=id_vocabulary,
        text_vocab=text_vocabulary,
        pos_vocab=part_of_speech_vocabulary,
        ner_vocab=named_entity_types_vocabulary
    )
    logger.info(f"Building datasets... done.")
    train_dataset_item = train_dataset[0]
    logger.info(f"id_ shape: {train_dataset_item.id_.shape}")
    logger.info(f"context shape: {train_dataset_item.context.shape}")
    logger.info(f"question shape: {train_dataset_item.question.shape}")
    logger.info(f"target shape: {train_dataset_item.target.shape}")
    logger.info(f"exact_match shape: {train_dataset_item.exact_match.shape}")
    logger.info(f"part_of_speech shape: {train_dataset_item.part_of_speech.shape}")
    logger.info(f"named_entity_type shape: {train_dataset_item.named_entity_type.shape}")
    logger.info(f"normalized_term_frequency shape: {train_dataset_item.normalized_term_frequency.shape}")

    logger.info("Building dataloaders...")
    collate_function = functools.partial(add_padding_and_batch_data,
                                         id_vocab=id_vocabulary,
                                         text_vocab=text_vocabulary,
                                         pos_vocab=part_of_speech_vocabulary,
                                         ner_vocab=named_entity_types_vocabulary,
                                         include_lengths=True,
                                         device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_function)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_function)
    logger.info("Building dataloaders... done.")
    for batch in train_dataloader:  # type: DrQATensorDatasetBatch
        logger.info("IDs:", batch.id_.shape)
        logger.info("Context:", batch.context[0].shape, batch.context[1].shape)
        logger.info("Question:", batch.question[0].shape, batch.question[1].shape)
        logger.info("Target:", batch.target.shape)
        logger.info("Exact match:", batch.exact_match.shape)
        logger.info("Part of speech:", batch.part_of_speech.shape)
        logger.info("Named entity type:", batch.named_entity_type.shape)
        logger.info("Normalized term frequency:", batch.normalized_term_frequency.shape)
        break

    logger.info("Loading embeddings...")
    glove_embeddings = load_glove_embeddings(path=args.embedding_path)
    embedding_matrix, found_indexes, not_found_indexes = extract_embeddings(
        embeddings=glove_embeddings,
        text_vocab=text_vocabulary,
        embedding_size=args.embedding_size
    )
    logger.info("Loading embeddings... done.")
    logger.info(f"Number of words found in GLoVE embeddings: {len(found_indexes)}/{len(text_vocabulary)} = "
          f"{100 * len(found_indexes) / len(text_vocabulary):.2f}%")
    logger.info(f"Number of words not found in GLoVE embeddings: {len(not_found_indexes)}/{len(text_vocabulary)} = "
          f"{100 * len(not_found_indexes) / len(text_vocabulary):.2f}%")

    logger.info("Building the model, optimizer, loss function and trainer...")
    padding_token_index = text_vocabulary.stoi(word=text_vocabulary.padding_token)
    model = DrQA(
        vocabulary_size=len(text_vocabulary),
        embedding_size=args.embedding_size,
        n_extra_features=4,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        padding_index=padding_token_index
    )
    model.load_word_embeddings(embedding_matrix=embedding_matrix, tune=True, found_indexes=found_indexes)
    model.to(device=device)
    logger.info(f"Number of parameters of the model: {model.count_parameters():,}")
    logger.info(f"Model architecture: {model}")
    optimizer = optim.Adamax(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_token_index)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        id_vocab=id_vocabulary,
        text_vocab=text_vocabulary,
        model_path=args.model_path
    )
    logger.info("Building the model, optimizer, loss function and trainer... done.")

    logger.info("Model training...")
    history = trainer.train(
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        n_epochs=args.n_epochs,
        gradient_clipping=args.gradient_clipping
    )
    logger.info(history)
    logger.info("Model Training... done.")
