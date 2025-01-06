import numpy as np
from typing import List, Tuple


def labels_to_midpoints(
    labels: np.ndarray, standard_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Converts an array of labels of intervals (min, max) to an array of midpoint values.

    labels: Array of labels representing indices of standard intervals.
    standard_intervals: List of predefined intervals, where each interval is a tuple (min, max).
    :returns: Array of midpoint values for each interval.
    """
    intervals = [standard_intervals[label] for label in labels]
    midpoints = [(interval[0] + interval[1]) / 2.0 for interval in intervals]
    return np.array(midpoints)