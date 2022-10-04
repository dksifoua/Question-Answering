import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .domain import DrQATensorDatasetBatch
from .utils import AverageMeter
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

    def train_step(self, loader: DataLoader, epoch: int, gradient_clipping: float) -> float:
        tracker = AverageMeter()
        self.model.train()
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for index, batch in pbar:  # type: int, DrQATensorDatasetBatch
            self.optimizer.zero_grad()
            starts, ends = self.model(batch)  # [batch_size, ctx_len]
            loss = self.criterion(starts, batch.target[:, 0]) + self.criterion(ends, batch.target[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=gradient_clipping)
            self.optimizer.step()
            tracker.update(loss.item())
            pbar.set_description(f"Epoch: {epoch + 1:02d} -     loss: {tracker.average:.3f}")
        return tracker.average

    def validate(self, loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        tracker, predictions = AverageMeter(), {}
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
            for _, batch in pbar:  # type: int, DrQATensorDatasetBatch
                starts, ends = self.model(batch)  # [batch_size, ctx_len]
                loss = self.criterion(starts, batch.target[:, 0]) + self.criterion(ends, batch.target[:, 1])
                start_indexes, end_indexes, _ = self.model.decode(
                    starts=F.softmax(starts, dim=-1),
                    ends=F.softmax(ends, dim=-1)
                )
                for index in range(starts.size(0)):
                    id_ = self.id_vocab.itos(batch.id_[index].item())
                    prediction = batch.context[0][index][start_indexes[index]:end_indexes[index] + 1]
                    predictions[id_] = ' '.join([self.text_vocab.itos(ind.item()) for ind in prediction])

                tracker.update(loss.item())
                pbar.set_description(f"Epoch: {epoch + 1:02d} - valid_loss: {tracker.average:.3f}")
        return tracker.average, predictions

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, n_epochs: int, gradient_clipping: float) \
            -> Dict[str, List[float]]:
        history, best_loss = {"loss": [], "valid_loss": [], "exact_match": [], "f1": []}, float('inf')
        for epoch in range(n_epochs):
            loss = self.train_step(loader=train_loader, epoch=epoch, gradient_clipping=gradient_clipping)
            valid_loss, predictions = self.validate(loader=valid_loader, epoch=epoch)

            history["loss"].append(loss)
            history["valid_loss"].append(valid_loss)

            if best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_path)
        return history
