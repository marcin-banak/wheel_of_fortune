from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.common.exceptions import MetricNotAvalible
from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum


@dataclass
class RegressionEvaluationResults(AbstractEvaluationResults):
    """
    Evaluation results for regression tasks.

    :param mae: The Mean Absolute Error.
    :param mse: The Mean Squared Error.
    :param rmse: The Root Mean Squared Error.
    :param r2: The R-squared metric.
    :param mape: The Mean Absolute Percentage Error.
    """

    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float

    IDEAL_METRICS = {
        MetricEnum.MAE: 0.0,
        MetricEnum.MSE: 0.0,
        MetricEnum.RMSE: 0.0,
        MetricEnum.R2: 1.0,
        MetricEnum.MAPE: 0.0,
    }

    def __init__(self, y_pred: pd.Series, y_test: pd.Series):
        """
        Initializes the evaluation results using predictions and ground truth values.

        :param y_pred: The predicted values.
        :param y_test: The ground truth values.
        """
        self.mae = mean_absolute_error(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(y_test, y_pred)
        self.mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    def get_metric(self, metric: MetricEnum):
        """
        Retrieves the value of a specific metric.

        :param metric: The metric to retrieve.
        :return: The value of the metric.
        :raises MetricNotAvalible: If the metric is not implemented.
        """
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
                return self.mape
            case MetricEnum.IDEAL_DISTANCE:
                return self.ideal_distance
        raise MetricNotAvalible(
            f"{metric} is not implemented for RegressionEvaluationResults"
        )

    def __str__(self):
        """
        Returns a string representation of the evaluation results.

        :return: A string containing formatted evaluation metrics.
        """
        return (
            f"MAE (Mean Absolute Error): {self.mae:.4f}\n"
            f"MSE (Mean Squared Error): {self.mse:.4f}\n"
            f"RMSE (Root Mean Squared Error): {self.rmse:.4f}\n"
            f"R2 (R-squared): {self.r2:.4f}\n"
            f"MAPE (Mean Absolute Percentage Error): {self.mape:.4f}\n"
            f"Ideal Distance: {self.ideal_distance:.4f}\n"
        )
