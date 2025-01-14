from evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum
from dataclasses import dataclass
from common.exceptions import MetricNotAvalible
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
)


@dataclass
class ClassificationEvaluationResults(AbstractEvaluationResults):
    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_classes_error: float

    IDEAL_METRICS = {
        MetricEnum.ACCURACY: 1.0,
        MetricEnum.PRECISION: 1.0,
        MetricEnum.RECALL: 1.0,
        MetricEnum.F1: 1.0
    }

    def __init__(self, y_pred: pd.Series, y_test: pd.Series):
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        self.f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        self.mean_classes_error = mean_absolute_error(y_test, y_pred)

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
            case MetricEnum.MEAN_CLASTERS_ERROR:
                return self.mean_classes_error
            case MetricEnum.IDEAL_DISTANCE:
                return self.ideal_distance
        raise MetricNotAvalible(f"{metric} is not implemented for ClassificationEvaluationResults")
    
    def __str__(self):
        return (
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1 Score: {self.f1:.4f}\n"
            f"  Mean Classes Error: {self.mean_classes_error:.4f}\n"
            f"  Ideal Distance: {self.ideal_distance:.4f}"
        )
