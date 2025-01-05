from dataclasses import asdict, field, dataclass

from xgboost import XGBClassifier

from models.AbstractModel import AbstractHyperparams, AbstractModel
from evaluation.evaluate_classification import (
    evaluate_classification,
    ClassificationEvaluationResults,
)

import numpy as np
from typing import List, Tuple


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


class PriceClassifierXGBoostModel(AbstractModel):
    """
    A custom classifier model based on XGBoost for price classification tasks.
    """

    def __init__(self, hyperparams: PriceClassifierXGBoostModelHyperparams):
        self.hyperparams = hyperparams
        self.model = XGBClassifier(
            **asdict(self.hyperparams),
            eval_metric="auc",
            enable_categorical=True,
            device="cuda",
        )

    def eval(self, y_pred: np.ndarray, y_test: np.ndarray) -> ClassificationEvaluationResults:
        return evaluate_classification(y_pred, y_test)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def _feature_importance(self) -> List[Tuple[str, float]]:
        return self.model.get_booster().get_score(importance_type="weight")
