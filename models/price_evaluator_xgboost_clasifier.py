from dataclasses import asdict, field, dataclass

from xgboost import XGBClassifier

from models.AbstractModel import AbstractHyperparams, AbstractModel
from evaluation.evaluate_classification import (
    evaluate_classification,
    ClassificationEvaluationResults,
)

import numpy as np


@dataclass
class PriceClassifierXGBoostModelHyperparams(AbstractHyperparams):
    learning_rate: float = field(metadata={"space": (0.01, 0.3), "type": "float"})
    reg_alpha: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    reg_lambda: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    max_depth: int = field(metadata={"space": (3, 15), "type": "int"})
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    min_child_weight: int = field(metadata={"space": (1, 10), "type": "int"})
    gamma: float = field(metadata={"space": (0.0, 5.0), "type": "float"})
    subsample: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
    colsample_bytree: float = field(metadata={"space": (0.5, 1.0), "type": "float"})


class PriceClassifierXGBoostModel(XGBClassifier, AbstractModel):
    """
    A custom classifier model based on XGBoost for price classification tasks.
    """

    def __init__(self, params: PriceClassifierXGBoostModelHyperparams):
        self.params = params
        super().__init__(
            **asdict(params),
            eval_metric="auc",
            tree_method="hist",
            enable_categorical=True,
            device="cuda"
        )

    def eval(self, y_pred: np.ndarray, y_test: np.ndarray) -> ClassificationEvaluationResults:
        return evaluate_classification(y_pred, y_test)
