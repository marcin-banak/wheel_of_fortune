from itertools import product
from typing import Callable, List, Dict, Any, Type
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from parameters.evaluation_results import EvaluationResults
from parameters.model_hyperparameters import ModelHyperparams


def cartesian_product(param_grid: Dict[str, List[Any]]) -> List[Dict[Any, Any]]:
    """
    Creates cartesian product of parameters.

    Args:
        param_grid (dict): Dictionary containing parameter names and lists of possible values.

    Returns:
        List[Dict]: List of variants of hyperparameters with their names.
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return param_combinations


def eval_model(
    model: Type,
    params: Dict[Any, Any],
    eval_function: Callable[[np.ndarray, np.ndarray], float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationResults:
    """
    Evaluates model's performance on given data using given evaluating function with given hyperparameters applied.

    Args:
        model (Type): The model class to instantiate.
        params (dict): Dictionary containing parameter names with their values.
        eval_function (callable): Function to evaluate predictions, accepting y_pred and y_test.
        X_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        X_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data labels.

    Returns:
        EvaluationResults: Evaluated score of the trained model.
    """
    model = model(**params)

    # Train the model with current parameters
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the predictions
    score = eval_function(y_pred, y_test)

    return score


def hyperparameter_tuning(
    model_class: Type,
    param_grid: Dict[str, List[Any]],
    eval_function: Callable[[np.ndarray, np.ndarray], float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparams_dataclass: Type[ModelHyperparams],
) -> ModelHyperparams:
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

    best_score = float("inf")
    best_params = None

    param_combinations = cartesian_product(param_grid)

    for i, params in enumerate(param_combinations):
        score = eval_model(
            model=model_class,
            params=params,
            eval_function=eval_function,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # Do przerobienia. Dorobić komparator
        if score < best_score:
            best_score = score
            best_params = params

        print(f"Evaluated {((i / len(param_combinations)) * 100):.2f}% of the params.")

    return hyperparams_dataclass(**best_params)


if __name__ == "__main__":

    def test_hyperparameter_tuning():
        from parameters.model_hyperparameters import XGBoostHyperparams

        def eval_function(y_pred, y_test):
            return mean_squared_error(y_test, y_pred)

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
            model_class=xgb.XGBRegressor,
            param_grid=param_grid,
            eval_function=eval_function,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparams_dataclass=XGBoostHyperparams,
        )

        print(best_hyperparams)

    test_hyperparameter_tuning()
