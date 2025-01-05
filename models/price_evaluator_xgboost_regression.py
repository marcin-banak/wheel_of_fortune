from dataclasses import asdict, field, dataclass

import numpy as np
from xgboost import XGBRegressor
from evaluation.evaluate_regression import (
    evaluate_regression,
    RegressionEvaluationResults,
)

from models.AbstractModel import AbstractHyperparams, AbstractModel


@dataclass
class PriceRegressorXGBoostModelHyperparams(AbstractHyperparams):
    learning_rate: float = field(metadata={"space": (0.01, 0.3), "type": "float"})
    reg_alpha: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    reg_lambda: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    max_depth: int = field(metadata={"space": (3, 15), "type": "int"})
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    min_child_weight: int = field(metadata={"space": (1, 10), "type": "int"})
    gamma: float = field(metadata={"space": (0.0, 5.0), "type": "float"})
    subsample: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
    colsample_bytree: float = field(metadata={"space": (0.5, 1.0), "type": "float"})


class PriceRegressorXGBoostModel(XGBRegressor, AbstractModel):
    """
    A custom regressor model based on XGBoost for price regression tasks.
    """

    def __init__(self, params: PriceRegressorXGBoostModelHyperparams):
        self.params = params
        super().__init__(**asdict(params))

    def eval(
        self, y_pred: np.ndarray, y_test: np.ndarray
    ) -> RegressionEvaluationResults:
        return evaluate_regression(y_pred, y_test)
