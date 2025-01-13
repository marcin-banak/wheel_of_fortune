from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum
from utils.calculate_ideal_distance import calculate_ideal_distance


@dataclass
class RegressionEvaluationResults(AbstractEvaluationResults):
    """
    Class to store and evaluate regression results.
    """

    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float

    IDEAL_METRICS = {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "mape": 0.0}

    @property
    def ideal_distance(self):
        return calculate_ideal_distance(
            self.IDEAL_METRICS,
            {"mae": self.mae, "rmse": self.rmse, "r2": self.r2, "mape": self.mape},
        )

    def get_metric(self, metric: MetricEnum):
        match metric:
            case MetricEnum.MAE:
                return self.mae
            case MetricEnum.MSE:
                return self.mse
            case MetricEnum.RMSE:
                return self.rmse
            case MetricEnum.R2:
                return self.r2
            case MetricEnum.MAPE:
                raise self.mape
            case MetricEnum.IDEAL_DISTANCE:
                return self.ideal_distance
        raise ValueError(f"{metric} is not implemented for RegressionEvaluationResults")

    def __str__(self):
        metrics_str = (
            f"  MAE (Mean Absolute Error): {self.mae:.4f}\n"
            f"  MSE (Mean Squared Error): {self.mse:.4f}\n"
            f"  RMSE (Root Mean Squared Error): {self.rmse:.4f}\n"
            f"  R2 (R-squared): {self.r2:.4f}\n"
            f"  MAPE (Mean Absolute Percentage Error): {self.mape:.4f}\n"
            f"  Ideal Distance: {self.ideal_distance:.4f}"
        )
        return metrics_str


def evaluate_regression(y_pred: np.ndarray, y_test: np.ndarray) -> RegressionEvaluationResults:
    """
    Calculates regression metrics for the given predicted and actual values.

    Metrics:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - R2: Coefficient of Determination
        - MAPE: Mean Absolute Percentage Error

    y_pred: Predicted values.
    y_test: Actual values.
    :returns: Results containing the computed regression metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return RegressionEvaluationResults(mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape)
