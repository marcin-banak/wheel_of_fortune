from dataclasses import dataclass, field
from typing import Optional, Union

from src.hyperparameters.AbstractHyperparameters import AbstractHyperparameters


@dataclass
class BasicClassificationHyperparameters(AbstractHyperparameters):
    """
    Basic hyperparameters for classification models.

    :param criterion: The function to measure the quality of a split.
    :param splitter: The strategy used to choose the split at each node.
    :param max_depth: The maximum depth of the tree.
    :param min_samples_split: The minimum number of samples required to split an internal node.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
    :param min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights.
    :param max_features: The number of features to consider when looking for the best split.
    :param max_leaf_nodes: The maximum number of leaf nodes in the tree.
    :param min_impurity_decrease: A node will be split if this split induces a decrease in impurity.
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
    """

    criterion: str = field(
        metadata={"space": ("gini", "entropy"), "type": "categorical"}
    )
    splitter: str = field(metadata={"space": ("best", "random"), "type": "categorical"})
    max_depth: Optional[int] = field(metadata={"space": (1, 20), "type": "int"})
    min_samples_split: int = field(metadata={"space": (2, 20), "type": "int"})
    min_samples_leaf: int = field(metadata={"space": (1, 10), "type": "int"})
    min_weight_fraction_leaf: float = field(
        metadata={"space": (0.0, 0.5), "type": "float"}
    )
    max_features: Optional[Union[int, float, str]] = field(
        metadata={"space": (None, "sqrt", "log2", 0.5), "type": "categorical"}
    )
    max_leaf_nodes: Optional[int] = field(metadata={"space": (2, 50), "type": "int"})
    min_impurity_decrease: float = field(
        metadata={"space": (0.0, 0.2), "type": "float"}
    )
    ccp_alpha: float = field(metadata={"space": (0.0, 1.0), "type": "float"})

    def __str__(self) -> str:
        """
        Returns a string representation of the hyperparameters.

        :return: A string containing formatted hyperparameter values.
        """
        return (
            f"criterion: {self.criterion}\n"
            f"splitter: {self.splitter}\n"
            f"max_depth: {self.max_depth}\n"
            f"min_samples_split: {self.min_samples_split}\n"
            f"min_samples_leaf: {self.min_samples_leaf}\n"
            f"min_weight_fraction_leaf: {self.min_weight_fraction_leaf}\n"
            f"max_features: {self.max_features}\n"
            f"max_leaf_nodes: {self.max_leaf_nodes}\n"
            f"min_impurity_decrease: {self.min_impurity_decrease}\n"
            f"ccp_alpha: {self.ccp_alpha}\n"
        )
