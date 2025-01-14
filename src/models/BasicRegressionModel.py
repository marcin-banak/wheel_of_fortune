from models.AbstractRegressionModel import AbstractRegressionModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from common.types import Params


class BasicRegressionModel(AbstractRegressionModel):
    def init_model(self, params: Params):
        self.model = RandomForestRegressor(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        self.model.predict(X_test)

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        self.model.set_params(hyperparameters_dict)
