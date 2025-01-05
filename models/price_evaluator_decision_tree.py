from dataclasses import asdict, field, dataclass
from typing import Optional, Union, List, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from evaluation.evaluate_classification import (
    evaluate_classification,
    ClassificationEvaluationResults,
)


from models.AbstractModel import AbstractHyperparams, AbstractModel


@dataclass
class DecisionTreeHyperparams(AbstractHyperparams):
    criterion: str = field(metadata={"space": ("gini", "entropy"), "type": "categorical"})
    splitter: str = field(metadata={"space": ("best", "random"), "type": "categorical"})
    max_depth: Optional[int] = field(metadata={"space": (1, 20), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 20), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 10), "type": "int"})
    min_weight_fraction_leaf: float = field(metadata={"space": (0.0, 0.5), "type": "float"})
    min_weight_fraction_leaf: float = field(metadata={"space": (0.0, 0.5), "type": "float"})
    max_features: Optional[Union[int, float, str]] = field(
        metadata={"space": (None, "sqrt", "log2", 0.5), "type": "categorical"}
    )
    max_leaf_nodes: Optional[int] = field(metadata={"space": (2, 50), "type": "int"})
    min_impurity_decrease: float = field(metadata={"space": (0.0, 0.2), "type": "float"})
    min_impurity_decrease: float = field(metadata={"space": (0.0, 0.2), "type": "float"})
    ccp_alpha: float = field(metadata={"space": (0.0, 1.0), "type": "float"})


class PriceClassifierBasicModel(AbstractModel):
    """
    A custom classifier model based on a Decision Tree for price classification tasks.
    """

    def __init__(self, hyperparams: DecisionTreeHyperparams):
        self.hyperparams = hyperparams
        self._feature_names = []
        self.model = DecisionTreeClassifier(**asdict(self.hyperparams))

    def eval(self, y_pred: np.ndarray, y_test: np.ndarray) -> ClassificationEvaluationResults:
        return evaluate_classification(y_pred, y_test)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._feature_names = X_train.columns.tolist()
        return self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def _feature_importance(self) -> List[Tuple[str, float]]:
        return list(zip(self._feature_names, map(float, self.model.feature_importances_)))
