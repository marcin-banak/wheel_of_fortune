from typing import List, Tuple
import numpy as np


def intervals_to_labels(
    tree_intervals: List[Tuple[float, float]],
    standard_intervals: List[Tuple[float, float]],
) -> np.array:
    """
    Maps intervals from `tree_intervals` to labels based on their index in `standard_intervals`.

    tree_intervals: A list of intervals that need to be mapped to labels.
    standard_intervals: A predefined list of intervals used for mapping. Each interval's index is treated as its label.
    :returns: An array of labels corresponding to the input intervals based on their index in `standard_intervals`.
    """
    return np.array([standard_intervals.index(interval) for interval in tree_intervals])


def labels_to_intervals(
    labels: np.array, standard_intervals: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Maps labels to their corresponding intervals based on a predefined list of standard intervals.

    labels: An array of labels to be converted to intervals.
    standard_intervals: A predefined list of intervals. Each label corresponds to the interval at the same index.
    :returns: A list of intervals corresponding to the input labels.
    """
    return [standard_intervals[label] for label in labels]
