from dataclasses import asdict, field, dataclass
from typing import Optional, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from evaluation.evaluate_classification import (
    evaluate_classification,
    ClassificationEvaluationResults,
)


from models.AbstractModel import AbstractHyperparams, AbstractModel


@dataclass
class DecisionTreeHyperparams(AbstractHyperparams):
    criterion: str = field(
        metadata={"space": ["gini", "entropy"], "type": "categorical"}
    )
    splitter: str = field(metadata={"space": ["best", "random"], "type": "categorical"})
    max_depth: Optional[int] = field(metadata={"space": (1, 20), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 20), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 10), "type": "int"})
    min_weight_fraction_leaf: float = field(
        metadata={"space": (0.0, 0.5), "type": "float"}
    )
    max_features: Optional[Union[int, float, str]] = field(
        metadata={"space": [None, "sqrt", "log2", 0.5], "type": "categorical"}
    )
    max_leaf_nodes: Optional[int] = field(metadata={"space": (2, 50), "type": "int"})
    min_impurity_decrease: float = field(
        metadata={"space": (0.0, 0.2), "type": "float"}
    )
    ccp_alpha: float = field(metadata={"space": (0.0, 1.0), "type": "float"})


class PriceClassifierBasicModel(DecisionTreeClassifier, AbstractModel):
    """
    A custom classifier model based on a Decision Tree for price classification tasks.
    """

    def __init__(self, params: DecisionTreeHyperparams):
        self.params = params
        super().__init__(**asdict(params))

    def eval(
        self, y_pred: np.ndarray, y_test: np.ndarray
    ) -> ClassificationEvaluationResults:
        return evaluate_classification(y_pred, y_test)
