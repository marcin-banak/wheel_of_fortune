from typing import Callable, Dict, Optional, Type, Union, List
import plotly.express as px
from training_process import training_process
from models.AbstractModel import AbstractHyperparams, AbstractModel
from evaluation.AbstractEvaluationResults import MetricEnum
from utils.cast_regression_to_classification import cast_regression_to_classification


def testing_intervals(
    model_name: str,
    model_class: Type[AbstractModel],
    hyperparameters_class: Type[AbstractHyperparams],
    metric: MetricEnum,
    intervals_functions: List[Callable[[float], float]] = None,
    const_params: Optional[Dict[str, Union[int, float, str]]] = None,
    sample: int = 0,
    category_encoding: bool = False,
    max_iters: int = 30,
    gpu_mode: bool = False,
    cv: int = 0,
    bootstraping_iters: int = 0,
):
    scores = []
    intervals_nums = []

    if "regressor" in model_name:
        for function in intervals_functions:
            score, intervals = cast_regression_to_classification(
                model_name, function, metric, category_encoding=category_encoding
            )
            scores.append(score)
            intervals_nums.append(len(intervals))

    else:
        for function in intervals_functions:
            score, intervals = training_process(
                model_name,
                model_class,
                hyperparameters_class,
                metric,
                function,
                const_params,
                sample,
                category_encoding,
                max_iters,
                gpu_mode,
                cv,
                bootstraping_iters,
            )
            scores.append(score)
            intervals_nums.append(len(intervals))

    sorted_data = sorted(zip(scores, intervals_nums), key=lambda x: x[1])
    sorted_scores, sorted_intervals_nums = zip(*sorted_data)

    metric_name = metric.name
    fig = px.scatter(
        x=sorted_intervals_nums,
        y=sorted_scores,
        labels={"y": f"{metric_name.capitalize()}", "x": "Number of Intervals"},
        title=f"Metric Score vs. Number of Intervals",
    )
    fig.update_traces(mode="markers+lines")
    fig.show()
