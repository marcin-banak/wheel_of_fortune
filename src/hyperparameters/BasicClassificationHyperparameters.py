from hyperparameters.AbstractHyperparameters import AbstractHyperparameters
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class BasicClassificationHyperparameters(AbstractHyperparameters):
    criterion: str = field(metadata={"space": ("gini", "entropy"), "type": "categorical"})
    splitter: str = field(metadata={"space": ("best", "random"), "type": "categorical"})
    max_depth: Optional[int] = field(metadata={"space": (1, 20), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 20), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 10), "type": "int"})
    min_weight_fraction_leaf: float = field(metadata={"space": (0.0, 0.5), "type": "float"})
    max_features: Optional[Union[int, float, str]] = field(
        metadata={"space": (None, "sqrt", "log2", 0.5), "type": "categorical"}
    )
    max_leaf_nodes: Optional[int] = field(metadata={"space": (2, 50), "type": "int"})
    min_impurity_decrease: float = field(metadata={"space": (0.0, 0.2), "type": "float"})
    ccp_alpha: float = field(metadata={"space": (0.0, 1.0), "type": "float"})
