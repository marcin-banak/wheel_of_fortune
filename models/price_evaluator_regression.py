from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


class PriceRegressorModel(XGBRegressor):
    """
    A custom regressor model based on XGBoost for price regression tasks.

    :param n_estimators: The number of trees in the ensemble. Default is 100.
    :param max_depth: The maximum depth of a tree. Default is 3.
    :param learning_rate: The step size shrinkage used to prevent overfitting. Default is 0.1.
    :param random_state: Seed used by the random number generator. Default is None.
    :param kwargs: Additional keyword arguments passed to the XGBRegressor.
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
            **kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regressor to the training data.

        :param X: The training input samples.
        :param y: The target values (regression labels).
        """

        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values for the input data.

        :param X: The input samples to predict.
        :return: The predicted continuous values.
        """

        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the mean squared error of the model on the given data.

        :param X: The input samples.
        :param y: The true values for the input samples.
        :return: The mean squared error of the model.
        """

        return mean_squared_error(y, self.predict(X))
