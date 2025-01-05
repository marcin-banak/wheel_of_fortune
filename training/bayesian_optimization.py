import sys
from dataclasses import fields
from pathlib import Path
from typing import Tuple, Type

import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from models.AbstractModel import AbstractHyperparams, AbstractModel

import json
from dataclasses import asdict
from utils.numpy_to_python import numpy_to_python

EXPORT_DIR = Path(__file__).parent.parent / "tuning_results"


def bayesian_optimization(
    model_name: str,
    model_class: Type[AbstractModel],
    hyperparam_class: Type[AbstractHyperparams],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 30,
    gpu_mode: bool = False,
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

    best_model = None
    best_score = None

    def create_param_space():
        dimensions = []

        for field in fields(hyperparam_class):
            if not isinstance(field.metadata.get("space"), tuple):
                raise ValueError(f"Field '{field.name}' must define 'space' metadata as a tuple.")

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
        nonlocal best_model, best_score

        params = {dim.name: param_values[i] for i, dim in enumerate(dimensions)}

        hyperparams = hyperparam_class(**params)
        print(f"Eval for {hyperparams}")
        model = model_class(hyperparams, gpu_mode)

        model.fit(X_train, y_train)
        evaluation_results = model.score(X_val, y_val)
        print(f"Actual score: {evaluation_results}")

        if best_score is None or best_score < evaluation_results:
            best_score = evaluation_results
            best_model = model

            with open(EXPORT_DIR / f"{model_name}.json", "w") as f:
                json.dump(asdict(best_model.hyperparams), f, indent=4, default=numpy_to_python)

        print(f"Best score: {best_score}")
        print(f"Best score hyperparams: {best_model.hyperparams}")

        return -evaluation_results.ideal_distance

    dimensions = create_param_space()

    gp_minimize(
        func=objective_function,
        dimensions=dimensions,
        n_calls=max_iter,
        random_state=42,
    )

    return best_model


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
