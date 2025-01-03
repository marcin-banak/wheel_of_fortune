from typing import Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class PriceClassifierDecisionTreeModel(DecisionTreeClassifier):
    """
    A custom classifier model based on a Decision Tree for price classification tasks.

    :param criterion: The function to measure the quality of a split. Default is 'gini'.
    :param splitter: The strategy used to split at each node. Default is 'best'.
    :param max_depth: The maximum depth of the tree. Default is None (unlimited depth).
    :param min_samples_split: The minimum number of samples required to split an internal node. Default is 2.
    :param min_samples_leaf: The minimum number of samples required to be at a leaf node. Default is 1.
    :param min_weight_fraction_leaf: The minimum weighted fraction of the sum of weights in a leaf node. Default is 0.0.
    :param max_features: The number of features to consider when looking for the best split. Default is None.
    :param max_leaf_nodes: The maximum number of leaf nodes. Default is None.
    :param min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
    :param ccp_alpha: Complexity parameter for Minimal Cost-Complexity Pruning. Default is 0.0.
    :param random_state: Seed used by the random number generator. Default is None.
    """

    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier to the training data.

        :param X: The training input samples.
        :param **kwargs:
        :param y: The target values (class labels).
        """
        super().fit(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class labels for the input data.

        :param X: The input samples to predict.
        :param **kwargs:
        :return: The predicted class labels.
        """
        return super().predict(X)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        Predict class probabilities for the input data if available.

        :param X: The input samples to predict probabilities for.
        :param **kwargs:
        :return: The predicted class probabilities, or None if not supported.
        """
        return super().predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Calculate the accuracy of the model on the given data.

        :param X: The input samples.
        :param y: The true labels for the input samples.
        :param **kwargs:
        :return: The accuracy score of the model.
        """
        return accuracy_score(y, self.predict(X))
