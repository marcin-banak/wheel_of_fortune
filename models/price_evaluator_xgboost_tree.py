from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class PriceClassifierXGBoostModel(XGBClassifier):
    """
    A custom classifier model based on XGBoost for price classification tasks.

    :param n_estimators: The number of trees in the ensemble. Default is 100.
    :param max_depth: The maximum depth of a tree. Default is 3.
    :param learning_rate: The step size shrinkage used to prevent overfitting. Default is 0.1.
    :param random_state: Seed used by the random number generator. Default is None.
    :param kwargs: Additional keyword arguments passed to the XGBClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier to the training data.

        :param X: The training input samples.
        :param y: The target values (class labels).
        """

        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        :param X: The input samples to predict.
        :return: The predicted class labels.
        """

        return super().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data.

        :param X: The input samples to predict probabilities for.
        :return: The predicted class probabilities.
        """

        return super().predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model on the given data.

        :param X: The input samples.
        :param y: The true labels for the input samples.
        :return: The accuracy score of the model.
        """

        return accuracy_score(y, self.predict(X))
