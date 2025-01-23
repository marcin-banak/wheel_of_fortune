from typing import Callable, List, Tuple

import pandas as pd


class IntervalsHandler:
    """
    Handles interval generation, classification, and reduction for data segmentation.

    :param data: Input data series to be classified.
    :param function: Function to calculate interval lengths.
    """

    def __init__(self, data: pd.Series, function: Callable[[float], float]):
        """
        Initializes the IntervalsHandler with data and a function for interval generation.

        :param data: Input data series.
        :param function: Function to calculate interval lengths based on a parameter.
        """
        self.data = data
        self.function = function
        self.intervals = self._generate_price_intervals(data.min(), data.max())
        self.intervals_number = len(self.intervals)

    def reduction(self):
        """
        Reduces the number of intervals based on the unique labels present in the data.
        Updates intervals and reindexes the data to match the reduced intervals.
        """
        used_labels = sorted(set(self.data))
        self.intervals = [self.intervals[i] for i in used_labels]

        old_to_new = {old: new for new, old in enumerate(used_labels)}
        self.data = pd.Series([old_to_new[label] for label in self.data])

    def classify(self):
        """
        Classifies the data into intervals.
        Updates the data to reflect the classified labels.
        """
        self.data = self._classify(self.data)

    def _classify(self, data: pd.Series) -> pd.Series:
        """
        Classifies the data into intervals.

        :param data: Data series to be classified.
        :return: A new series with classified labels.
        """

        def find_class(value):
            for idx, (lower, upper) in enumerate(self.intervals):
                if lower <= value <= upper:
                    return idx
            return self.intervals_number - 1

        return pd.Series(data.apply(find_class).astype(int))

    def _generate_price_intervals(self, A: int, B: int) -> List[Tuple[float, float]]:
        """
        Generates intervals between values A and B using the provided function.

        :param A: Start of the range.
        :param B: End of the range.
        :return: List of tuples representing intervals.
        """
        intervals = []
        start = A
        x = 0

        while start < B:
            length = int(self.function(x))
            if length <= 0:
                x += 1
                continue
            end = start + length
            if end > B:
                end = B
            intervals.append((start, end))
            start = end
            x += 1

        return intervals

    def regression_to_classification(self, y_pred: pd.Series) -> pd.Series:
        """
        Converts regression predictions into classification labels based on intervals.

        :param y_pred: Series of regression predictions.
        :return: Series of classification labels.
        """
        return self._classify(y_pred)
