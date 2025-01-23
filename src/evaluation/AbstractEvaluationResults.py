from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class MetricEnum(Enum):
    """
    Enumeration of available metrics for evaluation.
    """

    ACCURACY = 1
    PRECISION = 2
    RECALL = 3
    F1 = 4
    AUC_ROC = 5
    MEAN_CLASTERS_ERROR = 6
    MAE = 7
    MSE = 8
    RMSE = 9
    R2 = 10
    MAPE = 11
    IDEAL_DISTANCE = 12


METRIC_REVERSE_COMPARE = {
    MetricEnum.ACCURACY: False,
    MetricEnum.PRECISION: False,
    MetricEnum.RECALL: False,
    MetricEnum.F1: False,
    MetricEnum.AUC_ROC: False,
    MetricEnum.MEAN_CLASTERS_ERROR: True,
    MetricEnum.MAE: True,
    MetricEnum.MSE: True,
    MetricEnum.RMSE: True,
    MetricEnum.R2: False,
    MetricEnum.MAPE: True,
    MetricEnum.IDEAL_DISTANCE: True,
}


@dataclass
class AbstractEvaluationResults(ABC):
    """
    Abstract class for evaluation results.

    :param IDEAL_METRICS: A dictionary of ideal metric values.
    """

    IDEAL_METRICS: Dict[MetricEnum, float]

    @property
    def ideal_distance(self) -> float:
        """
        Calculates the sum of absolute differences between actual and ideal metric values.

        :return: The ideal distance.
        """
        return sum(
            abs(self.get_metric(metric) - ideal)
            for metric, ideal in self.IDEAL_METRICS.items()
        )

    @abstractmethod
    def get_metric(self, metric: MetricEnum) -> float:
        """
        Abstract method to retrieve the value of a specific metric.

        :param metric: The metric to retrieve.
        :return: The value of the metric.
        """
        raise NotImplementedError

    def get_metric_norm(self, metric: MetricEnum) -> float:
        """
        Normalizes a metric value based on its comparison direction.

        :param metric: The metric to normalize.
        :return: The normalized metric value.
        """
        return self.get_metric(metric) * (-1 if METRIC_REVERSE_COMPARE[metric] else 1)
