from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from dataclasses import dataclass
from dataclasses import fields
from typing import Type, Callable, Tuple
import numpy as np

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from models.AbstractModel import AbstractModel, AbstractHyperparams


def bayesian_optimization(
    model_class: Type[AbstractModel],
    hyperparam_class: Type[AbstractHyperparams],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 50,
) -> AbstractHyperparams:
    """
    Bayesian optimization for ML Model.

    :param model_class: Model's class.
    :param hyperparam_class: Hiperparameter class describing parameters space.
    :param X_train: Training data.
    :param y_train: Training data etiquetes.
    :param X_val: Validation data.
    :param y_val: Validation data etiquetes.
    :param max_iter: Maximal optimization iterations.
    :return: Optimal hiperparameter class.
    """

    def create_param_space():
        dimensions = []

        for field in fields(hyperparam_class):
            if not isinstance(field.metadata.get("space"), tuple):
                raise ValueError(
                    f"Field '{field.name}' must define 'space' metadata as a tuple."
                )

            param_range = field.metadata["space"]
            param_type = field.metadata.get("type", "float")

            if param_type == "float":
                dimensions.append(Real(*param_range, name=field.name))
            elif param_type == "int":
                dimensions.append(Integer(*param_range, name=field.name))
            elif param_type == "categorical":
                dimensions.append(Categorical(param_range, name=field.name))
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

        return dimensions

    def objective_function(param_values: Tuple):
        params = {dim.name: param_values[i] for i, dim in enumerate(dimensions)}

        hyperparams = hyperparam_class(**params)
        model = model_class(hyperparams)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        evaluation_results = model.eval(y_pred, y_val)

        return -evaluation_results.get_score()

    print("Started optimization.")

    dimensions = create_param_space()

    result = gp_minimize(
        func=objective_function,
        dimensions=dimensions,
        n_calls=max_iter,
        random_state=42,
    )

    best_params = {dim.name: result.x[i] for i, dim in enumerate(dimensions)}

    return hyperparam_class(**best_params)


if __name__ == "__main__":

    def test_bayesian_optimization():
        from models.price_evaluator_xgboost_clasifier import (
            PriceClassifierXGBoostModel,
            PriceClassifierXGBoostModelHyperparams,
        )

        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.rand(30, 10)
        y_val = np.random.randint(0, 3, 30)

        best_hyperparams = bayesian_optimization(
            model_class=PriceClassifierXGBoostModel,
            hyperparam_class=PriceClassifierXGBoostModelHyperparams,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_iter=11,
        )

        print(best_hyperparams)

    test_bayesian_optimization()
