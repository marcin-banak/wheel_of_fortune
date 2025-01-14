from hyperparameters.AbstractHyperparameters import AbstractHyperparameters
from dataclasses import dataclass, field


@dataclass
class BasicRegressionHyperparameters(AbstractHyperparameters):
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    max_depth: int = field(metadata={"space": (3, 30), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 10), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 5), "type": "int"})
    max_features: str = field(metadata={"space": ("sqrt", "log2"), "type": "categorical"})
    bootstrap: bool = field(metadata={"space": (True, False), "type": "categorical"})
