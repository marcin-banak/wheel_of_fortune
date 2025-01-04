from dataclasses import asdict, dataclass

import numpy as np
from xgboost import XGBRegressor
from evaluation.evaluate_regression import (
    evaluate_regression,
    RegressionEvaluationResults,
)

from models.AbstractModel import AbstractHyperparams, AbstractModel


@dataclass
class PriceRegressorXGBoostModelHyperparams(AbstractHyperparams):
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    n_estimators: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float


class PriceRegressorXGBoostModel(XGBRegressor, AbstractModel):
    """
    A custom regressor model based on XGBoost for price regression tasks.
    """

    def __init__(self, params: PriceRegressorXGBoostModelHyperparams):
        self.params = params
        super().__init__(
            **asdict(params),
            enable_categorical=True,
            device="cuda"
        )

    def eval(
        self, y_pred: np.ndarray, y_test: np.ndarray
    ) -> RegressionEvaluationResults:
        return evaluate_regression(y_pred, y_test)
