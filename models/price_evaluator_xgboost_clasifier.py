from dataclasses import asdict, dataclass

from xgboost import XGBClassifier

from models.AbstractModel import AbstractHyperparams, AbstractModel
from evaluation.evaluate_classification import (
    evaluate_classification,
    ClassificationEvaluationResults,
)

import numpy as np


@dataclass
class PriceClassifierXGBoostModelHyperparams(AbstractHyperparams):
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    n_estimators: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float


class PriceClassifierXGBoostModel(XGBClassifier, AbstractModel):
    """
    A custom classifier model based on XGBoost for price classification tasks.
    """

    def __init__(self, params: PriceClassifierXGBoostModelHyperparams):
        self.params = params
        super().__init__(
            **asdict(params),
            eval_metric="logloss",
            enable_categorical=True,
            device="cuda"
        )

    def eval(
        self, y_pred: np.ndarray, y_test: np.ndarray
    ) -> ClassificationEvaluationResults:
        return evaluate_classification(y_pred, y_test)
