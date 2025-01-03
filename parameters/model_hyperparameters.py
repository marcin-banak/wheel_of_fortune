from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ModelHyperparams:
    pass

@dataclass
class DecisionTreeHyperparams(ModelHyperparams):
    criterion: str              # "gini" or "entropy" for Gini/Information Gain for measuring quality of splits
    splitter: str               # "best" or "random" for feature split strategy
    max_depth: Optional[int]    # max tree depth
    min_samples_split: int      # minimum number of samples required to split an internal node
    min_samples_leaf: int       # minimum number of samples required to be at a leaf node
    min_weight_fraction_leaf: float # Minimum weighted fraction of the sum of weights in a leaf
    max_features: Optional[Union[int, float, str]] # Number of features to consider for the best split
    max_leaf_nodes: Optional[int]   # Grow a tree with max_leaf_nodes
    min_impurity_decrease: float    # Minimum impurity decrease for a split (only made if it decreases impurity more than this threshold)
    ccp_alpha: float = 0.0          # Complexity parameter for mccp (balancing complexity and accuracy)

@dataclass
class XGBoostHyperparams(ModelHyperparams):
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    n_estimators: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float
