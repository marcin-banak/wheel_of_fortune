from dataclasses import dataclass, field

from src.hyperparameters.AbstractHyperparameters import AbstractHyperparameters


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

    def __str__(self) -> str:
        return (
            f"learning_rate: {self.learning_rate}\n"
            f"reg_alpha: {self.reg_alpha}\n"
            f"reg_lambda: {self.max_depth}\n"
            f"max_depth: {self.max_depth}\n"
            f"n_estimators: {self.n_estimators}\n"
            f"min_child_weight: {self.min_child_weight}\n"
            f"gamma: {self.gamma}\n"
            f"subsample: {self.subsample}\n"
            f"colsample_bytree: {self.colsample_bytree}\n"
            f"max_delta_step: {self.max_delta_step}\n"
            f"colsample_bynode: {self.colsample_bynode}\n"
            f"colsample_bylevel: {self.colsample_bylevel}\n"
        )
