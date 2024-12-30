from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from parameters.evaluation_results import ClassificationEvaluationResults

import numpy as np


def evaluate_classification(
    y_pred: np.ndarray, y_test: np.ndarray
) -> ClassificationEvaluationResults:
    """
    Oblicza metryki klasyfikacyjne dla przewidywanych i rzeczywistych etykiet klas.

    Metryki:
        - Accuracy
        - Precision (średnia ważona)
        - Recall (średnia ważona)
        - F1-score (średnia ważona)

    Args:
        y_pred (np.ndarray): Przewidywane etykiety klas.
        y_test (np.ndarray): Rzeczywiste etykiety klas.

    Returns:
        ClassificationEvaluationResults: Wyniki obliczonych metryk.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return ClassificationEvaluationResults(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1
    )


def values_to_class_labels(values: np.ndarray, unique_intervals: np.ndarray) -> np.ndarray:
    """
    Przypisuje wartości do etykiet klas na podstawie unikalnych przedziałów.

    Args:
        values (np.ndarray): Tablica wartości liczbowych do przypisania.
        unique_intervals (np.ndarray): Tablica unikalnych przedziałów w postaci [(low1, high1), (low2, high2), ...].

    Returns:
        np.ndarray: Tablica etykiet klas odpowiadających przedziałom lub -1 dla wartości spoza zakresu.
    """
    labels = []
    for v in values:
        matched_label = -1
        for label, interval in enumerate(unique_intervals):
            low, high = interval
            if low <= v <= high:
                matched_label = label
                break
        labels.append(matched_label)
    return np.array(labels)
