from models.AbstractModel import AbstractModel
import numpy as np
from evaluation.AbstractEvaluationResults import MetricEnum
from sklearn.utils import resample


def bootstraping(
    model: AbstractModel,
    X: np.ndarray,
    y: np.ndarray,
    metric: MetricEnum,
    iters: int 
) -> float:
    results = []
    for i in range(iters):
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=i)
        model.fit(X_resampled, y_resampled)
        results.append(model.score(X, y).get_metric(metric))
    return np.array(results).mean()