from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evaluation.AbstractEvaluationResults import AbstractEvaluationResults


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

    def cost_function(self) -> float:
        """
        Computes a weighted cost function for regression metrics.
        Lower scores indicate a better model.

        :returns: The weighted sum of regression metrics.
        """
        return self.mae * 0.4 + self.rmse * 0.3 + self.mape * 0.2 + self.r2 * -0.2

    def __gt__(self, other) -> bool:
        """
        Compares two RegressionEvaluationResults objects using the cost function.

        other : Another RegressionEvaluationResults object to compare with.
        :returns: True if the current object has a lower cost function score, False otherwise.
        """
        if not isinstance(other, RegressionEvaluationResults):
            raise TypeError(
                "Comparison is only supported between RegressionEvaluationResults objects."
            )
        return self.cost_function() < other.cost_function()


def evaluate_regression(y_pred: np.ndarray, y_test: np.ndarray) -> RegressionEvaluationResults:
    """
    Calculates regression metrics for the given predicted and actual values.

    Metrics:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - R2: Coefficient of Determination
        - MAPE: Mean Absolute Percentage Error

    Args:
        y_pred: Predicted values.
        y_test: Actual values.

    Returns:
        RegressionEvaluationResults: Results containing the computed regression metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return RegressionEvaluationResults(mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape)


def labels_to_midpoints(
    labels: np.ndarray, standard_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Converts an array of labels of intervals (min, max) to an array of midpoint values.

    Args:
        labels: Array of labels representing indices of standard intervals.
        standard_intervals: List of predefined intervals, where each interval is a tuple (min, max).

    Returns:
        np.ndarray: Array of midpoint values for each interval.
    """
    intervals = [standard_intervals[label] for label in labels]
    midpoints = [(interval[0] + interval[1]) / 2.0 for interval in intervals]
    return np.array(midpoints)
