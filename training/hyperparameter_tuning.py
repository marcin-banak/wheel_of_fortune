from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np

from evaluation.AbstractEvaluationResults import AbstractEvaluationResults
from models.AbstractModel import AbstractHyperparams, AbstractModel
from utils.cartesian_product import cartesian_product


def hyperparameter_tuning(
    model_class: Type[AbstractModel],
    hyperparams_class: Type[AbstractHyperparams],
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> AbstractModel:
    """
    Performs hyperparameter tuning using Cartesian product of parameter values.

    Args:
        model_class (Type): The model class to instantiate.
        param_grid (dict): Dictionary containing parameter names and lists of possible values.
        eval_function (callable): Function to evaluate predictions, accepting y_pred and y_test.
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data labels.
        hyperparams_dataclass (Type[ModelHyperparams]): Dataclass representing the hyperparameters.

    Returns:
        ModelHyperparams: The best combination of hyperparameters as a dataclass.
    """

    best_results = None
    best_model = None

    param_combinations = cartesian_product(param_grid)

    for i, params_data in enumerate(param_combinations):
        hyperparams = hyperparams_class(**params_data)
        model = model_class(hyperparams)
        model.fit(X_train, y_train)
        results = model.score(X_test, y_test)

        if not best_results or results < best_results:
            best_results = results
            best_model = model

        print(f"Evaluated {((i / len(param_combinations)) * 100):.2f}% of the params.")

    return best_model


if __name__ == "__main__":

    from sklearn.metrics import mean_squared_error

    from models.price_evaluator_xgboost_regression import (
        PriceRegressorXGBoostModel,
        PriceRegressorXGBoostModelHyperparams,
    )

    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 200],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 1.0, 2.0, 5.0],
    }

    X_train, X_test = np.random.rand(100, 10), np.random.rand(20, 10)
    y_train, y_test = np.random.rand(100), np.random.rand(20)

    best_hyperparams = hyperparameter_tuning(
        model_class=PriceRegressorXGBoostModel,
        hyperparams_dataclass=PriceRegressorXGBoostModelHyperparams,
        param_grid=param_grid,
        eval_function=mean_squared_error,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print(best_hyperparams)
