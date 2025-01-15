from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum
from typing import List


def metric_list(score_list: List[AbstractEvaluationResults], metric: MetricEnum) -> List[float]:
    return [score.get_metric(metric) for score in score_list]
