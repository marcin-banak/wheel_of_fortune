import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.common.types import Params
from src.models.AbstractRegressionModel import AbstractRegressionModel
from src.hyperparameters.BasicRegressionHyperparameters import BasicRegressionHyperparameters


class BasicRegressionModel(AbstractRegressionModel):
    HYPERPARAMETERS_CLASS = BasicRegressionHyperparameters

    def _init_model(self, params: Params):
        self.model = RandomForestRegressor(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X_test))

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        self.model.set_params(**hyperparameters_dict)
