from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from parameters.evaluation_results import RegressionEvaluationResults

import numpy as np


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
        y_pred (np.ndarray): Predicted values.
        y_test (np.ndarray): Actual values.

    Returns:
        RegressionEvaluationResults: Results containing the computed regression metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return RegressionEvaluationResults(mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape)


def intervals_to_midpoints(intervals: np.ndarray) -> np.ndarray:
    """
    Converts an array of intervals (min, max) to an array of midpoint values.

    Args:
        intervals (np.ndarray): Array of tuples [(min1, max1), (min2, max2), ...].

    Returns:
        np.ndarray: Array of midpoint values for each interval.
    """
    midpoints = [(interval[0] + interval[1]) / 2.0 for interval in intervals]
    return np.array(midpoints)
