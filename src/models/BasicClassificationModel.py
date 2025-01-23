import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.common.types import Params
from src.hyperparameters.BasicClassificationHyperparameters import (
    BasicClassificationHyperparameters,
)
from src.models.AbstractClassificationModel import AbstractClassificationModel


class BasicClassificationModel(AbstractClassificationModel):
    """
    Basic classification model using a decision tree.

    Inherits from AbstractClassificationModel and implements basic functionality
    for classification tasks.

    :param HYPERPARAMETERS_CLASS: Class defining the hyperparameters structure.
    """

    HYPERPARAMETERS_CLASS = BasicClassificationHyperparameters

    def _init_model(self, params: Params):
        """
        Initializes the DecisionTreeClassifier with the provided parameters.

        :param params: Dictionary of parameters to initialize the model.
        """
        self.model = DecisionTreeClassifier(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the model to the training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions on the test data.

        :param X_test: Test features.
        :return: Predicted labels as a pandas Series.
        """
        return pd.Series(self.model.predict(X_test))

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        """
        Updates the hyperparameters of the model.

        :param hyperparameters_dict: Dictionary of hyperparameter values.
        """
        self.model.set_params(**hyperparameters_dict)
