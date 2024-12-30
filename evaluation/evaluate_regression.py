from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from parameters.evaluation_results import RegressionEvaluationResults

import numpy as np

def evaluate_regression(y_pred: np.ndarray, y_test: np.ndarray) -> RegressionEvaluationResults:
    """
    Oblicza metryki regresyjne dla podanych przewidywanych i rzeczywistych wartości.

    Metryki:
        - MAE: Średni błąd bezwzględny
        - MSE: Średni błąd kwadratowy
        - RMSE: Pierwiastek z MSE
        - R2: Współczynnik determinacji
        - MAPE: Średni procentowy błąd bezwzględny

    Args:
        y_pred (np.ndarray): Przewidywane wartości.
        y_test (np.ndarray): Rzeczywiste wartości.

    Returns:
        RegressionEvaluationResults: Wyniki obliczeń zawierające metryki regresji.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return RegressionEvaluationResults(
        mae=mae, 
        mse=mse, 
        rmse=rmse, 
        r2=r2, 
        mape=mape
    )

def interval_to_midpoint(intervals: np.ndarray) -> np.ndarray:
    """
    Zamienia tablicę przedziałów (min, max) na tablicę wartości środkowych.

    Args:
        intervals (np.ndarray): Tablica krotek [(min1, max1), (min2, max2), ...].

    Returns:
        np.ndarray: Tablica wartości środkowych dla każdego przedziału.
    """
    midpoints = [(interval[0] + interval[1]) / 2.0 for interval in intervals]
    return np.array(midpoints)