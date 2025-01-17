import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.common.types import Params
from src.hyperparameters.BasicClassificationHyperparameters import (
    BasicClassificationHyperparameters,
)
from src.models.AbstactClassificationModel import AbstractClassificationModel


class BasicClassificationModel(AbstractClassificationModel):
    HYPERPARAMETERS_CLASS = BasicClassificationHyperparameters

    def _init_model(self, params: Params):
        self.model = DecisionTreeClassifier(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X_test))

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        self.model.set_params(**hyperparameters_dict)
