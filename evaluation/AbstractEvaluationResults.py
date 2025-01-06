from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class MetricEnum(Enum):
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
    @abstractmethod
    def get_metric(metric: MetricEnum) -> float:
        NotImplementedError

    def compare(self, other: AbstractEvaluationResults, metric: MetricEnum) -> bool:
        val = self.get_metric(metric)
        other_val = other.get_metric(metric)
        return val < other_val if METRIC_REVERSE_COMPARE[metric] else val > other_val
