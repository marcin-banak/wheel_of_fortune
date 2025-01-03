from typing import List, Tuple
import numpy as np


def evaluate_model_consistency(
    regression_predictions: np.ndarray,
    tree_labels: np.ndarray,
    standard_intervals: List[Tuple[float, float]],
):
    """
    Checks the consistency of linear regression predictions with the intervals predicted by decision trees.

    Parameters:
        regression_predictions: List of point values predicted by linear regression.
        tree_intervals: List of intervals labels predicted by decision trees.

    Returns:
        dict: A dictionary containing the percentage of consistency, average distance from the interval, and interval widths.
    """
    if len(regression_predictions) != len(tree_labels):
        raise ValueError(
            "The length of the prediction list and the interval list must be the same."
        )

    tree_intervals = [standard_intervals[label] for label in tree_labels]
    total_cases = len(regression_predictions)
    consistent_count = 0
    total_distance = 0

    for prediction, (lower, upper) in zip(regression_predictions, tree_intervals):
        if lower <= prediction <= upper:
            consistent_count += 1
        else:
            if prediction < lower:
                total_distance += lower - prediction
            elif prediction > upper:
                total_distance += prediction - upper

    consistency_percentage = (consistent_count / total_cases) * 100

    inconsistent_cases = total_cases - consistent_count
    average_distance = (
        total_distance / inconsistent_cases if inconsistent_cases > 0 else 0
    )

    return {
        "consistency_percentage": consistency_percentage,
        "average_distance": average_distance,
    }
