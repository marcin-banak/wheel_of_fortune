from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
)

from evaluation.AbstractEvaluationResults import AbstractEvaluationResults
from utils.calculate_ideal_distance import calculate_ideal_distance
from dataclasses import dataclass
import numpy as np
from evaluation.AbstractEvaluationResults import MetricEnum

from sklearn.metrics import roc_auc_score


@dataclass
class ClassificationEvaluationResults(AbstractEvaluationResults):
    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_classes_error: float
    roc_auc: float = None

    IDEAL_METRICS = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

    @property
    def ideal_distance(self):
        metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        return calculate_ideal_distance(self.IDEAL_METRICS, metrics)
    
    def get_metric(self, metric: MetricEnum):
        match metric:
            case MetricEnum.ACCURACY:
                return self.accuracy
            case MetricEnum.PRECISION:
                return self.precision
            case MetricEnum.RECALL:
                return self.recall
            case MetricEnum.F1:
                return self.f1
            case MetricEnum.AUC_ROC:
                return self.roc_auc
            case MetricEnum.MEAN_CLASTERS_ERROR:
                return self.mean_classes_error
            case MetricEnum.IDEAL_DISTANCE:
                return self.ideal_distance
        raise ValueError(f"{metric} is not implemented for ClassificationEvaluationResults")

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
    mean_classes_error = mean_absolute_error(y_test, y_pred)

    if y_pred_proba is not None:
        # Handle binary and multi-class cases
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(
                y_test, y_pred_proba, multi_class="ovr", average="weighted"
            )
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        roc_auc = None

    return ClassificationEvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_classes_error=mean_classes_error,
        roc_auc=roc_auc,
    )



