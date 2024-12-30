from dataclasses import dataclass



@dataclass
class EvaluationResults:
    pass


@dataclass
class ClassificationEvaluationResults(EvaluationResults):
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class RegressionEvaluationResults(EvaluationResults):
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float
