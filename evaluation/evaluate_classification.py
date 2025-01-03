from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parameters.evaluation_results import ClassificationEvaluationResults
from typing import List, Tuple

import numpy as np


def evaluate_classification(
    y_pred: np.ndarray, y_test: np.ndarray
) -> ClassificationEvaluationResults:
    """
    Calculates classification metrics for predicted and actual class labels.

    Metrics:
        - Accuracy
        - Precision (weighted average)
        - Recall (weighted average)
        - F1-score (weighted average)

    Args:
        y_pred: Predicted class labels.
        y_test: Actual class labels.

    Returns:
        ClassificationEvaluationResults: Results of computed metrics.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return ClassificationEvaluationResults(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1
    )


def values_to_class_labels(
    values: np.ndarray, standard_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Assigns values to class labels based on unique intervals.

    Args:
        values: Array of numerical values to be assigned.
        standard_intervals: Array of unique intervals in the form [(low1, high1), (low2, high2), ...].

    Returns:
        np.ndarray: Array of class labels corresponding to the intervals or -1 for values out of range.
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
