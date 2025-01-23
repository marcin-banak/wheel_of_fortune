from dataclasses import dataclass, field

from src.hyperparameters.AbstractHyperparameters import AbstractHyperparameters


@dataclass
class BasicRegressionHyperparameters(AbstractHyperparameters):
    """
    Basic hyperparameters for regression models.

    :param n_estimators: Number of trees in the ensemble.
    :param max_depth: Maximum depth of the tree.
    :param min_samples_split: Minimum number of samples required to split an internal node.
    :param min_samples_leaf: Minimum number of samples required to be at a leaf node.
    :param max_features: Number of features to consider when looking for the best split.
    :param bootstrap: Whether bootstrap samples are used when building trees.
    """

    n_estimators: int = field(metadata={"space": (50, 500), "type": "int"})
    max_depth: int = field(metadata={"space": (3, 30), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 10), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 5), "type": "int"})
    max_features: str = field(
        metadata={"space": ("sqrt", "log2"), "type": "categorical"}
    )
    bootstrap: bool = field(metadata={"space": (True, False), "type": "categorical"})

    def __str__(self) -> str:
        """
        Returns a string representation of the hyperparameters.

        :return: A string containing formatted hyperparameter values.
        """
        return (
            f"n_estimators: {self.n_estimators}\n"
            f"max_depth: {self.max_depth}\n"
            f"min_samples_split: {self.min_samples_split}\n"
            f"min_samples_leaf: {self.min_samples_leaf}\n"
            f"max_features: {self.max_features}\n"
            f"bootstrap: {self.bootstrap}\n"
        )
