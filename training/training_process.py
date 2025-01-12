from typing import Callable, Dict, Optional, Type, Union, Tuple, List

import pandas as pd

from data_processing.dtype_mapping import dtype_mapping
from models.AbstractModel import AbstractHyperparams, AbstractModel
from training.bayesian_optimization import bayesian_optimization
from utils.class_reduction import class_reduction
from utils.classification_range_generator import generate_price_intervals
from utils.classify import classify
from utils.encode_categorial_columns import encode_categorical_columns
from utils.export_model import save_model
from evaluation.AbstractEvaluationResults import MetricEnum
from training.train import train
from dataclasses import asdict


def training_process(
    model_name: str,
    model_class: Type[AbstractModel],
    hyperparameters_class: Type[AbstractHyperparams],
    metric: MetricEnum,
    intervals_function: Optional[Callable[[float], float]] = None,
    const_params: Optional[Dict[str, Union[int, float, str]]] = None,
    sample: int = 0,
    category_encoding: bool = False,
    max_iters: int = 30,
    gpu_mode: bool = False,
    cv: int = 0,
    bootstraping_iters: int = 0,
) -> Tuple[float, List[float]]:
    data = pd.read_csv("../data/processed_car_sale_ads.csv", low_memory=False)
    data = dtype_mapping(data)

    if sample:
        data = data.sample(n=sample)

    if category_encoding:
        encode_categorical_columns(data)

    X = data.iloc[:, 1:]
    y = data["Price"]

    intervals = []

    if intervals_function:
        intervals = generate_price_intervals(y.min(), y.max(), intervals_function)
        y = classify(y, intervals)
        y, intervals = class_reduction(y, intervals)
        print("Price intervals:")
        [print(f"{int(interval[0])} - {int(interval[1])}") for interval in intervals]

    params = const_params or asdict(bayesian_optimization(
        model_name,
        model_class,
        hyperparameters_class,
        metric,
        X,
        y,
        max_iters,
        gpu_mode,
        cv,
    ))

    model = model_class(hyperparameters_class(**params))
    score = train(model, X, y, metric, cv, bootstraping_iters)

    save_model(model, model_name)

    print(model.hyperparams)
    print(f"{metric.name.capitalize()}: {score}")
    model.feature_importance()

    return score, intervals
