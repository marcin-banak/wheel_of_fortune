from models.AbstractModel import AbstractModel
import numpy as np
from evaluation.AbstractEvaluationResults import MetricEnum
from utils.custom_cross_validation import custom_cross_validation
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils.bootstraping import bootstraping


def train(
    model: AbstractModel,
    X: np.ndarray,
    y: np.ndarray,
    metric: MetricEnum,
    cv: int = 0,
    bootstraping_iters: int = 0,
) -> float:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)

    if bootstraping_iters > 0:
        return bootstraping(model, X, y, metric, bootstraping_iters)
    if cv > 0:
        return custom_cross_validation(model, X, y, metric, cv)
    return model.score(X_test, y_test).get_metric(metric)
