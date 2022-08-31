from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from .utils import metrics


class Trainer:

    def __int__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, loader: DataLoader, epoch: int, gradient_clipping: float) -> float:
        pass

    def validate(self, loader: DataLoader, epoch: int) -> Tuple[float, Tensor]:
        pass

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, n_epochs: int, gradient_clipping: float) \
            -> Dict[str, List[float]]:
        history, best_loss = {"loss": [], "valid_loss": [], "exact_match": [], "f1": []}, float('inf')
        for epoch in range(n_epochs):
            loss = self.train_step(loader=train_loader, epoch=epoch, gradient_clipping=gradient_clipping)
            valid_loss, predictions = self.validate(loader=valid_loader, epoch=epoch)
            exact_match, f1 = metrics(predictions=predictions, qas=None)  # To be replaced

            history["loss"].append(loss)
            history["valid_loss"].append(valid_loss)
            history["exact_match"].append(exact_match)
            history["f1"].append(f1)

            if best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), './checkpoints/DrQA.pth')

        return history