from models.AbstactClassificationModel import AbstractClassificationModel
from common.types import Params
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class BasicClassificationModel(AbstractClassificationModel):
    def init_model(self, params: Params):
        self.model = DecisionTreeClassifier(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        self.model.predict(X_test)

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        self.model.set_params(hyperparameters_dict)
