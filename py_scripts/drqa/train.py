import argparse

import torch.nn as nn
import torch.optim as optim

from question_answering.drqa.model import DrQA
from question_answering.trainer import Trainer
from question_answering.vocabulary import Vocabulary
from question_answering.configuration import Configuration


if __name__ == "__main__":
    configuration = Configuration()
    parser = argparse.ArgumentParser(description="Train a DrQA model")
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
                        type=int,
                        required=False,
                        default=configuration.base.training.drqa.learning_rate,
                        help="Number of epochs")
    parser.add_argument("--n_epochs",
                        action="store",
                        type=int,
                        required=False,
                        default=configuration.base.training.drqa.n_epochs,
                        help="Number of epochs")
    parser.add_argument("--gradient_clipping",
                        action="store",
                        type=int,
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

    print("Loading vocabularies...")
    text_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("text"))
    part_of_speech_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("part_of_speech"))
    named_entity_types_vocabulary = Vocabulary.load(path=args.vocabulary_uncompleted_path.format("named_entity_types"))
    print("Loading vocabularies... ok")
    print(f"Length of text vocabulary: {len(text_vocabulary):,}")
    print(f"Length of part of speech vocabulary: {len(part_of_speech_vocabulary):,}")
    print(f"Length of named entity type vocabulary: {len(named_entity_types_vocabulary):,}")

    print("Building the model, optimizer, loss function and trainer...")
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
    print(f"Number of parameters of the model: {model.count_parameters():,}")
    print(f"Model architecture: {model}")
    optimizer = optim.Adamax(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_token_index)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        id_vocab=Vocabulary(padding_token="", unknown_token=""),
        text_vocab=text_vocabulary,
        model_path=args.model_path
    )
    print("Building the model, optimizer, loss function and trainer... ok")
