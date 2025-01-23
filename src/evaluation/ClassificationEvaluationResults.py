from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

from src.common.exceptions import MetricNotAvalible
from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum


@dataclass
class ClassificationEvaluationResults(AbstractEvaluationResults):
    """
    Evaluation results for classification tasks.

    :param accuracy: The accuracy score.
    :param precision: The precision score.
    :param recall: The recall score.
    :param f1: The F1 score.
    :param mean_classes_error: The mean absolute error of classes.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    mean_classes_error: float

    IDEAL_METRICS = {
        MetricEnum.ACCURACY: 1.0,
        MetricEnum.PRECISION: 1.0,
        MetricEnum.RECALL: 1.0,
        MetricEnum.F1: 1.0,
    }

    def __init__(self, y_pred: pd.Series, y_test: pd.Series):
        """
        Initializes the evaluation results using predictions and ground truth values.

        :param y_pred: The predicted values.
        :param y_test: The ground truth values.
        """
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        self.recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        self.f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        self.mean_classes_error = mean_absolute_error(y_test, y_pred)

    def get_metric(self, metric: MetricEnum):
        """
        Retrieves the value of a specific metric.

        :param metric: The metric to retrieve.
        :return: The value of the metric.
        :raises MetricNotAvalible: If the metric is not implemented.
        """
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
        raise MetricNotAvalible(
            f"{metric} is not implemented for ClassificationEvaluationResults"
        )

    def __str__(self):
        """
        Returns a string representation of the evaluation results.

        :return: A string containing formatted evaluation metrics.
        """
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1 Score: {self.f1:.4f}\n"
            f"Mean Classes Error: {self.mean_classes_error:.4f}\n"
            f"Ideal Distance: {self.ideal_distance:.4f}\n"
        )
