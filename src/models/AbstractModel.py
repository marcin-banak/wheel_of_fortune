from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from datetime import datetime
from typing import Tuple, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import resample
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from src.common.Config import Config
from src.common.exceptions import (
    HyperparameterSpaceNotDefined,
    NoHyperparametersFile,
    NoModelFile,
    NotCompatibleHyperparametersModel,
)
from src.common.types import Params
from src.common.json_dump import json_dump
from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum
from src.hyperparameters.AbstractHyperparameters import AbstractHyperparameters
from src.utils.stratified_resample import stratified_resample
from src.utils.plot import plot


class AbstractModel(ABC):
    hyperparameters: AbstractHyperparameters
    gpu_mode: bool

    HYPERPARAMETERS_CLASS: Type[AbstractHyperparameters]
    CPU_MODE_PARAMS: Params = {}
    GPU_MODE_PARAMS: Params = {}

    def __init__(self, hyperparameters_dict: Params = {}, gpu_mode: bool = False):
        self.gpu_mode = gpu_mode
        if hyperparameters_dict:
            self.set_hyperparameters(hyperparameters_dict)
        self._init_model({
            **hyperparameters_dict, 
            **(self.GPU_MODE_PARAMS if gpu_mode else self.CPU_MODE_PARAMS)
        })

    @abstractmethod
    def _init_model(self, params: Params):
        NotImplementedError

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        NotImplementedError

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        NotImplementedError

    @abstractmethod
    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> AbstractEvaluationResults:
        NotImplementedError

    @abstractmethod
    def _set_model_hyperparameters(self, hyperparameters_dict: Params):
        NotImplementedError

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> AbstractEvaluationResults:
        return self.eval(self.predict(X_test), y_test)

    def set_hyperparameters(self, hyperparameters_dict: Params):
        self.hyperparameters = self.HYPERPARAMETERS_CLASS(**hyperparameters_dict)
        self._set_model_hyperparameters(hyperparameters_dict)

    def save_model(self, name: str):
        Config.saved_models_dir.mkdir(exist_ok=True, parents=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = Config.saved_models_dir / f"{name}_{current_time}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(name: str) -> AbstractModel:
        model_path = Config.trained_models_dir / f"{name}.pkl"
        if not model_path.exists():
            raise NoModelFile(f"There is no model file {model_path}")
        with open(model_path, "r") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    
    def save_hyperparameters(self, name: str):
        Config.saved_hyperparameters_dir.mkdir(exist_ok=True, parents=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Config.saved_hyperparameters_dir / f"{name}_{current_time}.json"
        json_dump({"model": type(self).__name__, **asdict(self.hyperparameters)}, output_path)

    def load_hyperparameters(self, name: str):
        hyperparameters_path = Config.optimized_hyperparameters_dir / f"{name}.json"
        if not hyperparameters_path.exists():
            raise NoHyperparametersFile(f"There is no hyperparameters file {hyperparameters_path}")
        with open(hyperparameters_path, "rb") as f:
            loaded_data = json.load(f)
        loaded_data_model = loaded_data.pop("model", None)
        if not loaded_data_model or loaded_data_model != type(self).__name__:
            raise NotCompatibleHyperparametersModel(
                f"Loaded hyperparameters are compatible with {loaded_data_model},"
                f"but not with {type(self).__name__}"
            )
        self.set_hyperparameters(loaded_data)

    def bootstraping(self, X: pd.DataFrame, y: pd.Series, iters: int, metric: MetricEnum) -> float:
        results = []
        for i in range(iters):
            X_resampled, y_resampled = stratified_resample(X, y, random_state=i)
            self.fit(X_resampled, y_resampled)
            results.append(self.score(X.to_numpy(), y.to_numpy()).get_metric(metric))
        return np.array(results).mean()

    def cross_validation(self, X: pd.DataFrame, y: pd.Series, cv: int, metric: MetricEnum) -> float:
        results = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.fit(X_train, y_train)
            results.append(self.score(X_test, y_test).get_metric(metric))
        return np.array(results).mean()

    def bayesian_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric: MetricEnum,
        max_iters: int = 10,
        verbose: bool = False
    ):
        best_hyperparameters = None
        best_norm = None
        norms = []

        def objective_function(param_values: Tuple):
            nonlocal best_norm, best_hyperparameters, norms

            params = {dim.name: param_values[i] for i, dim in enumerate(dimensions)}
            self.set_hyperparameters(params)
            if verbose:
                print(f"Eval for:\n{self.hyperparameters}")
            self.fit(X_train, y_train)
            results = self.score(X_test, y_test)
            if verbose:
                print(f"Actual results:\n{results}")
            norm = results.get_metric_norm(metric)
            norms.append(norm)
            if best_norm is None or best_norm < norm:
                best_norm = norm
                best_hyperparameters = self.hyperparameters
                self.save_hyperparameters(f"{type(self).__name__}_bayesian")
            return -norm

        dimensions = []

        for field in fields(self.HYPERPARAMETERS_CLASS):
            if not isinstance(field.metadata.get("space"), tuple):
                raise HyperparameterSpaceNotDefined(
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

        gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            n_calls=max_iters,
            random_state=42,
        )

        plot(
            [i for i, _ in enumerate(norms)],
            {metric.name: norms},
            f"{metric.name} by time",
            "Iteration",
            metric.name,
        )

        self.set_hyperparameters(asdict(best_hyperparameters))
