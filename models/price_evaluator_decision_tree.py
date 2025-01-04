from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from models.AbstractModel import AbstractHyperparams, AbstractModel


@dataclass
class DecisionTreeHyperparams(AbstractHyperparams):
    criterion: str  # "gini" or "entropy" for Gini/Information Gain for measuring quality of splits
    splitter: str  # "best" or "random" for feature split strategy
    max_depth: Optional[int]  # max tree depth
    min_samples_split: int  # minimum number of samples required to split an internal node
    min_samples_leaf: int  # minimum number of samples required to be at a leaf node
    min_weight_fraction_leaf: float  # Minimum weighted fraction of the sum of weights in a leaf
    max_features: Optional[
        Union[int, float, str]
    ]  # Number of features to consider for the best split
    max_leaf_nodes: Optional[int]  # Grow a tree with max_leaf_nodes
    min_impurity_decrease: float  # Minimum impurity decrease for a split (only made if it decreases impurity more than this threshold)
    ccp_alpha: float = 0.0  # Complexity parameter for mccp (balancing complexity and accuracy)


class PriceClassifierBasicModel(DecisionTreeClassifier, AbstractModel):
    """
    A custom classifier model based on a Decision Tree for price classification tasks.
    """

    def __init__(self, params: DecisionTreeHyperparams):
        self.params = params
        super().__init__(**asdict(params))
