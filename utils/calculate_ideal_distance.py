import numpy as np


def calculate_ideal_distance(IDEAL_METRICS: dict, metrics: dict) -> float:
    """
    Calculate distance from ideal metrics for each key in the dictionaries.

    IDEAL_METRICS: Dictionary of ideal metric values.
    metrics: Dictionary of actual metric values.
    :returns: Euclidean distance between actual metrics and ideal metrics.
    """
    if set(IDEAL_METRICS.keys()) != set(metrics.keys()):
        raise ValueError("The keys in IDEAL_METRICS and metrics must match.")

    distance = np.sqrt(sum((metrics[key] - IDEAL_METRICS[key]) ** 2 for key in IDEAL_METRICS))
    return distance
