import numpy as np
from models.AbstractModel import AbstractModel
from evaluation.AbstractEvaluationResults import MetricEnum, METRIC_REVERSE_COMPARE
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


def custom_cross_validation(
    model: AbstractModel,
    X: np.ndarray,
    y: np.ndarray,
    metric: MetricEnum,
    cv: int
) -> float:
    custom_scorer = make_scorer(
        lambda y_test, y_pred: model.eval(y_pred, y_test).get_metric(metric),
        greater_is_better=not METRIC_REVERSE_COMPARE[metric],
    )
    return cross_validate(model.model, X, y, cv=cv, scoring=custom_scorer)["test_score"].mean()
