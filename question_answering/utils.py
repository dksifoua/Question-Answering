import re
import string
import collections
from typing import List, Tuple


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
    num_same = sum(common.values())
    f1_score = 0
    if num_same != 0:
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1_score = (2 * precision * recall) / (precision + recall)

    return prediction == ground_truth, f1_score


def max_metrics_over_ground_truths(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    scores = [get_scores(prediction, ground_truth) for ground_truth in ground_truths]
    em_score = max(scores, key=lambda score: score[0])[0]
    f1_score = max(scores, key=lambda score: score[1])[1]
    return em_score, f1_score


def metrics(predictions: dict, qas) -> Tuple[float, float]:
    pass
