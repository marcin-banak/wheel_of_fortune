from dataclasses import dataclass

import pandas as pd

from src.utils.IntervalsHandler import IntervalsHandler


@dataclass
class Consistency:
    """
    Represents the consistency metrics for regression and classification predictions.

    :param consistency_percentage: The percentage of consistent predictions.
    :param average_distance: The average distance of inconsistent predictions.
    """

    consistency_percentage: float
    average_distance: float

    def __init__(
        self,
        regression_pred: pd.Series,
        classification_pred: pd.Series,
        intervals_handler: IntervalsHandler,
    ):
        """
        Initializes the consistency metrics based on regression and classification predictions.

        :param regression_pred: The regression predictions.
        :param classification_pred: The classification predictions.
        :param intervals_handler: The handler for intervals corresponding to classification labels.
        """
        classifier_intervals = [
            intervals_handler.intervals[label] for label in classification_pred
        ]

        total_cases = len(regression_pred)
        consistent_count = 0
        total_distance = 0

        for prediction, (lower, upper) in zip(regression_pred, classifier_intervals):
            if lower <= prediction <= upper:
                consistent_count += 1
            else:
                if prediction < lower:
                    total_distance += lower - prediction
                elif prediction > upper:
                    total_distance += prediction - upper

        self.consistency_percentage = (consistent_count / total_cases) * 100
        inconsistents = total_cases - consistent_count
        self.average_distance = (
            total_distance / inconsistents if inconsistents > 0 else 0
        )
