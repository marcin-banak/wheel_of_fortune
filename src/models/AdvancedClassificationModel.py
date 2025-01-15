import pandas as pd
from xgboost import XGBClassifier

from src.common.types import Params
from src.models.AbstactClassificationModel import AbstractClassificationModel
from src.hyperparameters.AdvancedClassificationHyperparameters import AdvancedClassificationHyperparameters


class AdvancedClassificationModel(AbstractClassificationModel):
    HYPERPARAMETERS_CLASS = AdvancedClassificationHyperparameters
    CPU_RUN_PARAMS = {"tree_method": "hist", "n_jobs": -1}
    GPU_RUN_PARAMS = {"tree_method": "gpu_hist", "gpu_id": 0}

    def _init_model(self, params: Params):
        self.model = XGBClassifier(**params, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return self.model.predict(X_test)

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        self.model.set_params(hyperparameters_dict)
