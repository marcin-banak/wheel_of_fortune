from __future__ import annotations


from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from evaluation.AbstractEvaluationResults import AbstractEvaluationResults

from utils.calculate_ideal_distance import calculate_ideal_distance
from dataclasses import dataclass
import numpy as np


from sklearn.metrics import roc_auc_score


@dataclass
class ClassificationEvaluationResults(AbstractEvaluationResults):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float = None

    IDEAL_METRICS = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}

    @property
    def ideal_distance(self):
        metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }
        return calculate_ideal_distance(self.IDEAL_METRICS, metrics)

    def get_score(self):
        score = (
            0.4 * self.accuracy
            + 0.35 * self.precision
            + 0.15 * self.recall
            + 0.1 * self.f1
            + 0.1 * self.roc_auc
        )
        return score

    def __gt__(self, other: ClassificationEvaluationResults) -> bool:
        if not isinstance(other, ClassificationEvaluationResults):
            raise TypeError(
                "Comparison is only supported between ClassificationEvaluationResults objects."
            )
        return self.ideal_distance < other.ideal_distance


def evaluate_classification(
    y_pred: np.ndarray, y_test: np.ndarray, y_pred_proba: np.ndarray = None
) -> ClassificationEvaluationResults:
    """
    Calculates classification metrics for predicted and actual class labels.

    Metrics:
        - Accuracy
        - Precision (weighted average)
        - Recall (weighted average)
        - F1-score (weighted average)
        - ROC AUC (weighted average for multi-class, binary otherwise)

    y_pred: Predicted class labels.
    y_test: Actual class labels.
    y_pred_proba: Predicted probabilities for ROC AUC calculation (required for ROC AUC).
    :returns: Results of computed metrics.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if y_pred_proba is not None:
        # Handle binary and multi-class cases
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = None

    return ClassificationEvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
    )


def values_to_class_labels(
    values: np.ndarray, standard_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Assigns values to class labels based on unique intervals.

    values: Array of numerical values to be assigned.
    standard_intervals: Array of unique intervals in the form [(low1, high1), (low2, high2), ...].
    :returns: Array of class labels corresponding to the intervals or -1 for values out of range.
    """
    labels = []
    for v in values:
        matched_label = -1
        for label, interval in enumerate(standard_intervals):
            low, high = interval
            if low <= v <= high:
                matched_label = label
                break
        labels.append(matched_label)
    return np.array(labels)
