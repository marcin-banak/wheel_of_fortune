import pandas as pd
from xgboost import XGBRegressor

from src.common.Config import Config
from src.common.types import Params
from src.hyperparameters.AdvancedRegressionHyperparameters import AdvancedRegressionHyperparameters
from src.models.AbstractRegressionModel import AbstractRegressionModel


class AdvancedRegressionModel(AbstractRegressionModel):
    """
    Advanced regression model using XGBoost.

    Inherits from AbstractRegressionModel and implements functionality for regression tasks.

    :param HYPERPARAMETERS_CLASS: Class defining the hyperparameters structure.
    :param CPU_RUN_PARAMS: Parameters for running the model on CPU.
    :param GPU_RUN_PARAMS: Parameters for running the model on GPU.
    """

    HYPERPARAMETERS_CLASS = AdvancedRegressionHyperparameters
    CPU_RUN_PARAMS = {"tree_method": "hist", "n_jobs": -1}
    GPU_RUN_PARAMS = {"tree_method": "gpu_hist", "gpu_id": 0}

    def _init_model(self, params: Params):
        """
        Initializes the XGBoost model with the provided parameters.

        :param params: Dictionary of parameters to initialize the model.
        """
        self.model = XGBRegressor(**params, random_state=42)

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
        :return: Predicted values as a pandas Series.
        """
        return pd.Series(self.model.predict(X_test))

    def export_xgboost_instance(self, name: str):
        """
        Exports the XGBoost model instance to a file in UBJ format.

        :param name: Name of the model file.
        """
        Config.saved_models_dir.mkdir(exist_ok=True, parents=True)
        models_path = Config.saved_models_dir / f"{name}.ubj"
        self.model.save_model(models_path)

    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        """
        Updates the hyperparameters of the model.

        :param hyperparameters_dict: Dictionary of hyperparameter values.
        """
        self.model.set_params(**hyperparameters_dict)
