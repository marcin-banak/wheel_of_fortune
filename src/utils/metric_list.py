from typing import List

from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum


def metric_list(score_list: List[AbstractEvaluationResults], metric: MetricEnum) -> List[float]:
    return [score.get_metric(metric) for score in score_list]
