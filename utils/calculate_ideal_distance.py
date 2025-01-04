import numpy as np


IDEAL_METRICS = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0}


def calculate_ideal_distance(metrics) -> float:
    """Calculate distance from ideal for every metric and sum up"""

    ideal_metrics = list(IDEAL_METRICS.values())
    distance = np.sqrt(np.sum((np.array(metrics) - np.array(ideal_metrics)) ** 2))
    return distance
