from dataclasses import dataclass


@dataclass
class ClassificationEvaluationResults:
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class RegressionEvaluationResults:
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float
