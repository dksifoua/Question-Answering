import re
import torch
import random
import string
import collections
import numpy as np
from typing import Dict, List, Tuple

from question_answering.domain import RawDatasetItem


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False


def normalize(answer: str) -> str:
    """Performs a series of cleaning steps on the ground truth and predicted answer."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def get_scores(prediction: str, ground_truth: str) -> Tuple[float, float]:
    prediction, ground_truth = normalize(prediction), normalize(ground_truth)
    prediction_tokens, ground_truth_tokens = prediction.split(), ground_truth.split()

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    number_same = sum(common.values())
    f1_score = 0
    if number_same != 0:
        precision = 1.0 * number_same / len(prediction_tokens)
        recall = 1.0 * number_same / len(ground_truth_tokens)
        f1_score = (2 * precision * recall) / (precision + recall)

    return prediction == ground_truth, f1_score


def max_metrics_over_ground_truths(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    scores = [get_scores(prediction, ground_truth) for ground_truth in ground_truths]
    em_score = max(scores, key=lambda score: score[0])[0]
    f1_score = max(scores, key=lambda score: score[1])[1]
    return em_score, f1_score


def metrics(predictions: Dict[str, str], qas: List[RawDatasetItem]) -> Tuple[float, float]:
    ground_truths = collections.defaultdict(lambda: [])
    for qa in qas:
        if qa.id_ in predictions:
            ground_truths[qa.id_].append(qa.answer.text)

    em_scores, f1_scores, total = [], [], 0
    for id_ in predictions:
        em_score, f1_score = max_metrics_over_ground_truths(predictions[id_], ground_truths[id_])
        em_scores.append(em_score)
        f1_scores.append(f1_score)
        total += 1

    em_score = 100.0 * sum(em_scores) / total
    f1_score = 100.0 * sum(f1_scores) / total
    return em_score, f1_score


class AverageMeter:
    # TODO
    #  Add typing

    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
