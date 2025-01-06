from typing import Callable, Dict, List, Optional, Type, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing.dtype_mapping import dtype_mapping
from models.AbstractModel import AbstractHyperparams, AbstractModel
from training.bayesian_optimization import bayesian_optimization
from training.hyperparameter_tuning import hyperparameter_tuning
from utils.class_reduction import class_reduction
from utils.classification_range_generator import generate_price_intervals
from utils.classify import classify
from utils.encode_categorial_columns import encode_categorical_columns
from utils.export_model import save_model
from evaluation.AbstractEvaluationResults import MetricEnum


def training_process(
    model_name: str,
    model_class: Type[AbstractModel],
    hyperparameters_class: Type[AbstractHyperparams],
    metric: MetricEnum,
    intervals_function: Optional[Callable[[float], float]] = None,
    param_grid: Optional[Dict[str, List[Union[int, float, str]]]] = None,
    sample: int = 0,
    category_encoding: bool = False,
    max_iters: int = 30,
    gpu_mode: bool = False,
):
    data = pd.read_csv("../data/processed_car_sale_ads.csv", low_memory=False)
    data = dtype_mapping(data)
    data.head(10)

    if sample:
        data = data.sample(n=sample)

    if category_encoding:
        encode_categorical_columns(data)

    X = data.iloc[:, 1:]
    y = data["Price"]

    if intervals_function:
        intervals = generate_price_intervals(y.min(), y.max(), intervals_function)
        y = classify(y, intervals)
        y, intervals = class_reduction(y, intervals)
        [print(f"{int(interval[0])} - {int(interval[1])}") for interval in intervals]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    if param_grid:
        best_model = hyperparameter_tuning(
            model_class,
            hyperparameters_class,
            metric,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            gpu_mode,
        )
    else:
        best_model = bayesian_optimization(
            model_name,
            model_class,
            hyperparameters_class,
            metric,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iters,
            gpu_mode,
        )

    save_model(best_model, model_name)

    print(best_model.hyperparams)
    print(best_model.score(X_test, y_test))
    print(best_model.feature_importance())
