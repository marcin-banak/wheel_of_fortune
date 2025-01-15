from dataclasses import dataclass, field

from src.hyperparameters.AbstractHyperparameters import AbstractHyperparameters


@dataclass
class BasicRegressionHyperparameters(AbstractHyperparameters):
    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    max_depth: int = field(metadata={"space": (3, 30), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 10), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 5), "type": "int"})
    max_features: str = field(metadata={"space": ("sqrt", "log2"), "type": "categorical"})
    bootstrap: bool = field(metadata={"space": (True, False), "type": "categorical"})

    def __str__(self) -> str:
        return (
            f"n_estimators: {self.n_estimators}\n"
            f"max_depth: {self.max_depth}\n"
            f"min_sample_split: {self.min_samples_split}\n"
            f"min_samples_leaf: {self.min_samples_leaf}\n"
            f"max_features: {self.max_features}\n"
            f"bootstrap: {self.bootstrap}\n"
        )
