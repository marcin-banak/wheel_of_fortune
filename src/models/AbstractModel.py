from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from common.Config import Config
from common.exceptions import NoModelFile, NoHyperparametersFile, NotCompatibleHyperparametersModel
from datetime import datetime
import pickle
from evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum
from hyperparameters.AbstractHyperparameters import AbstractHyperparameters
import json
from dataclasses import asdict
from typing import Type, Tuple
from common.types import Params
from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import KFold
from skopt import gp_minimize
from utils.create_hyperparams_space import create_hyperparams_space


class AbstractModel(ABC):
    hyperparameters: AbstractHyperparameters
    gpu_mode: bool

    HYPERPARAMETERS_CLASS: Type[AbstractHyperparameters]
    CPU_MODE_PARAMS: Params = {"n_jobs": -1}
    GPU_MODE_PARAMS: Params = {}

    def __init__(self, hyperparameters_dict: Params, gpu_mode: bool = False):
        self.gpu_mode = gpu_mode
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
        Config.saved_models_dir.parent.mkdir(exist_ok=True, parents=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = Config.saved_models_dir / f"{name}_{current_time}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(name: str) -> AbstractModel:
        model_path = Config.trained_models_dir / f"{name}.pkl"
        if not model_path.exists():
            raise NoModelFile(f"There is no model file {model_path}")
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    
    def save_hyperparameters(self, name: str):
        Config.saved_hyperparameters_dir.parent.mkdir(exist_ok=True, parents=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hyperparameters_path = Config.saved_hyperparameters_dir / f"{name}_{current_time}.pkl"
        with open(hyperparameters_path, "wb") as f:
            json.dump({"model": type(self).__name__, **asdict(self.hyperparams)}, f, indent=4)

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
            X_resampled, y_resampled = resample(X, y, replace=True, random_state=i)
            self.fit(X_resampled, y_resampled)
            results.append(self.score(X, y).get_metric(metric))
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
        max_iters: int = 10
    ):
        best_hyperparameters = None
        best_norm = None

        def objective_function(param_values: Tuple):
            nonlocal best_norm, best_hyperparameters

            params = {dim.name: param_values[i] for i, dim in enumerate(dimensions)}
            self.set_hyperparameters(params)
            print(f"Eval for {self.hyperparameters}")
            self.fit(X_train, y_train)
            results = self.score(X_test, y_test)
            print(f"Actual results: {results}")
            norm = results.get_metric_norm(metric)
            if best_norm is None or best_norm < norm:
                best_norm = norm
                best_hyperparameters = self.hyperparameters
                self.save_hyperparameters(f"{type(self).__name__}_bayesian")
            return norm

        dimensions = create_hyperparams_space(self.HYPERPARAMETERS_CLASS)

        gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            n_calls=max_iters,
            random_state=42,
        )

        self.set_hyperparameters(best_hyperparameters)
