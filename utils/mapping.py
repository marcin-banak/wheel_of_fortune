from typing import List, Tuple
import numpy as np


def intervals_to_labels(
    tree_intervals: List[Tuple[float, float]], standard_intervals: List[Tuple[float, float]]
) -> np.array:
    """
    Maps intervals to labels based on their index.
    """
    return np.array([standard_intervals.index(interval) for interval in tree_intervals])


def labels_to_intervals(
    labels: np.array, standard_intervals: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Maps labels to their corresponding intervals based on a predefined list of standard intervals.
    """
    return [standard_intervals[label] for label in labels]
