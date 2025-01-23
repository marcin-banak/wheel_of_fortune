from typing import List

from src.evaluation.AbstractEvaluationResults import AbstractEvaluationResults, MetricEnum


def metric_list(
    score_list: List[AbstractEvaluationResults], metric: MetricEnum
) -> List[float]:
    """
    Extracts a list of specific metric values from a list of evaluation results.

    :param score_list: List of evaluation results.
    :param metric: The metric to extract from each evaluation result.
    :return: List of metric values.
    """
    return [score.get_metric(metric) for score in score_list]
