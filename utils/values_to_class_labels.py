import numpy as np
from typing import List, Tuple


def values_to_class_labels(
    values: np.ndarray, standard_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Assigns values to class labels based on unique intervals.

    values: Array of numerical values to be assigned.
    standard_intervals: Array of unique intervals in the form [(low1, high1), (low2, high2), ...].
    :returns: Array of class labels corresponding to the intervals or -1 for values out of range.
    """

    labels = []
    for v in values:
        matched_label = -1
        for label, interval in enumerate(standard_intervals):
            low, high = interval
            if low <= v <= high:
                matched_label = label
                break
        labels.append(matched_label)
    return np.array(labels)
