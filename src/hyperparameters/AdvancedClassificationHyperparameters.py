from hyperparameters.AbstractHyperparameters import AbstractHyperparameters
from dataclasses import dataclass, field


@dataclass
class AdvancedClassificationHyperparameters(AbstractHyperparameters):
    learning_rate: float = field(metadata={"space": (0.01, 0.3), "type": "float"})
    reg_alpha: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    reg_lambda: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    max_depth: int = field(metadata={"space": (3, 15), "type": "int"})
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    min_child_weight: int = field(metadata={"space": (1, 10), "type": "int"})
    gamma: float = field(metadata={"space": (0.0, 5.0), "type": "float"})
    subsample: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
    colsample_bytree: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
    max_delta_step: float = field(metadata={"space": (0.0, 10.0), "type": "float"})
    colsample_bynode: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
    colsample_bylevel: float = field(metadata={"space": (0.5, 1.0), "type": "float"})
