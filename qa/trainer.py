import tqdm
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from .domain import DrQATensorDatasetBatch
from .utils import AverageMeter, metrics
from .vocabulary import Vocabulary


class Trainer:

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, id_vocab: Vocabulary,
                 text_vocab: Vocabulary, model_path: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.id_vocab = id_vocab
        self.text_vocab = text_vocab
        self.model_path = model_path

    def get_predictions(self, batch: DrQATensorDatasetBatch, starts: Tensor, ends: Tensor) -> Dict[str, str]:
        start_indexes, end_indexes, _ = self.model.decode(
            starts=F.softmax(starts, dim=-1),
            ends=F.softmax(ends, dim=-1)
        )

        predictions = {}
        for index in range(len(start_indexes)):
            id_ = self.id_vocab.itos(batch.id_[index].item())
            prediction = batch.context[0][index][start_indexes[index]:end_indexes[index] + 1]
            predictions[id_] = ' '.join([self.text_vocab.itos(ind.item()) for ind in prediction])

        return predictions

    def compute_metrics_and_update_tracker(self, batch: DrQATensorDatasetBatch, starts: Tensor, ends: Tensor,
                                           tracker: AverageMeter, loader: DataLoader) -> None:
        # TODO
        #   Make batch type general not specific since the trainer will be use for different types of models
        predictions = self.get_predictions(batch=batch, starts=starts, ends=ends)
        em, f1 = metrics(predictions=predictions, qas=loader.dataset.data)

        tracker.update(key="em", value=em)
        tracker.update(key="f1", value=f1)

    def train_step(self, loader: DataLoader, epoch: int, gradient_clipping: float) -> AverageMeter:
        tracker = AverageMeter(keys=["loss", "em", "f1"])
        self.model.train()
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for index, batch in pbar:  # type: int, DrQATensorDatasetBatch
            self.optimizer.zero_grad()
            starts, ends = self.model(batch)  # [batch_size, ctx_len]
            loss = self.criterion(starts, batch.target[:, 0]) + self.criterion(ends, batch.target[:, 1])
            loss.backward()
            if gradient_clipping is not None:
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=gradient_clipping)
            self.optimizer.step()
            tracker.update(key="loss", value=loss.item())
            self.compute_metrics_and_update_tracker(
                batch=batch,
                starts=starts,
                ends=ends,
                tracker=tracker,
                loader=loader
            )
            pbar.set_description(
                f"Epoch: {epoch + 1:03d} |       loss: {tracker.average['loss']:6.3f} |       "
                f"em: {tracker.average['em']:6.3f} |       f1: {tracker.average['f1']:6.3f}"
            )
            # TODO
            #   Only break in for testing the training class (unit test or over fit a single batch)
            break
        return tracker

    def validate(self, loader: DataLoader) -> AverageMeter:
        tracker = AverageMeter(keys=["loss", "em", "f1"])
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
            for _, batch in pbar:  # type: int, DrQATensorDatasetBatch
                starts, ends = self.model(batch)  # [batch_size, ctx_len]
                loss = self.criterion(starts, batch.target[:, 0]) + self.criterion(ends, batch.target[:, 1])
                tracker.update(key="loss", value=loss.item())
                self.compute_metrics_and_update_tracker(
                    batch=batch,
                    starts=starts,
                    ends=ends,
                    tracker=tracker,
                    loader=loader
                )
                pbar.set_description(
                    f"           | valid_loss: {tracker.average['loss']:6.3f} | valid_em: {tracker.average['em']:6.3f} "
                    f"| valid_f1: {tracker.average['f1']:6.3f}"
                )
                # TODO
                #   Only break in for testing the training class (unit test or over fit a single batch)
                break
        return tracker

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, n_epochs: int, gradient_clipping: float) \
            -> Dict[str, List[float]]:
        history = {
            "loss": [], "valid_loss": [],
            "em": [], "valid_em": [],
            "f1": [], "valid_f1": []
        }
        # best_loss = float('inf')
        for epoch in range(n_epochs):
            print(
                "|----------------------------------------------------------------------------------------------------"
                "---------|"
            )
            train_tracker = self.train_step(loader=train_loader, epoch=epoch, gradient_clipping=gradient_clipping)
            valid_tracker = self.validate(loader=valid_loader)

            history["loss"].append(train_tracker.average["loss"])
            history["valid_loss"].append(valid_tracker.average["loss"])
            history["em"].append(train_tracker.average["em"])
            history["valid_em"].append(valid_tracker.average["loss"])
            history["f1"].append(train_tracker.average["f1"])
            history["valid_f1"].append(valid_tracker.average["f1"])

        return history
