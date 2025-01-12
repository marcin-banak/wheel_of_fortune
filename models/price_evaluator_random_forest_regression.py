from dataclasses import asdict, dataclass, field
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from evaluation.evaluate_regression import (
    RegressionEvaluationResults,
    evaluate_regression,
)
from models.AbstractModel import AbstractHyperparams, AbstractModel

@dataclass
class PriceRegressorRandomForestHyperparams(AbstractHyperparams):
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    max_depth: int = field(metadata={"space": (3, 30), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 10), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 5), "type": "int"})
    max_features: str = field(
        metadata={
            "space": ("auto", "sqrt", "log2"),
            "type": "categorical",
        }
    )
    bootstrap: bool = field(
        metadata={"space": (True, False), "type": "categorical"}
    )


class PriceRegressorRandomForestModel(AbstractModel):
    """
    A custom regressor model based on RandomForest for price regression tasks.
    """

    def __init__(
        self, hyperparams: PriceRegressorRandomForestHyperparams
    ):
        self.hyperparams = hyperparams
        self.model = RandomForestRegressor(
            **asdict(self.hyperparams),
            n_jobs=-1,
            random_state=42
        )

    def eval(
        self, y_pred: np.ndarray, y_test: np.ndarray
    ) -> RegressionEvaluationResults:
        return evaluate_regression(y_pred, y_test)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def _feature_importance(self) -> Dict[str, float]:
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))
