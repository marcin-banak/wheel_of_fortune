from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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
        super().__init__(**asdict(params))
