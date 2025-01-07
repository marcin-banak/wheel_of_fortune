from pathlib import Path
from typing import Tuple, Type

import numpy as np
from skopt import gp_minimize
from utils.create_hyperparams_space import create_hyperparams_space
from models.AbstractModel import AbstractHyperparams, AbstractModel

import json
from dataclasses import asdict
from utils.numpy_to_python import numpy_to_python
from evaluation.AbstractEvaluationResults import MetricEnum, METRIC_REVERSE_COMPARE
from training.train import train

EXPORT_DIR = Path(__file__).parent.parent / "tuning_results"


def bayesian_optimization(
    model_name: str,
    model_class: Type[AbstractModel],
    hyperparams_class: Type[AbstractHyperparams],
    metric: MetricEnum,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 30,
    gpu_mode: bool = False,
    cv: int = 0
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
    best_results = None

    def objective_function(param_values: Tuple):
        nonlocal best_model, best_results

        params = {dim.name: param_values[i] for i, dim in enumerate(dimensions)}
        hyperparams = hyperparams_class(**params)
        model = model_class(hyperparams, gpu_mode)
        
        print(f"Eval for {hyperparams}")
        results = train(model, X, y, metric, cv)
        print(f"Actual results: {results}")
        results *= (-1 if METRIC_REVERSE_COMPARE[metric] else 1)

        if best_results is None or best_results < results:
            best_results = results
            best_model = model

            with open(EXPORT_DIR / f"{model_name}.json", "w") as f:
                json.dump(
                    asdict(best_model.hyperparams), f, indent=4, default=numpy_to_python
                )

        print(f"Best hyperparams: {best_model.hyperparams}")
        return results

    dimensions = create_hyperparams_space(hyperparams_class)

    gp_minimize(
        func=objective_function,
        dimensions=dimensions,
        n_calls=max_iter,
        random_state=42,
    )

    return best_model.hyperparams
