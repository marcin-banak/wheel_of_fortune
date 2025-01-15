from typing import Callable, List, Tuple

import pandas as pd


class IntervalsHandler:
    def __init__(self, data: pd.Series, function: Callable[[float], float]):
        self.data = data
        self.function = function
        self.intervals = self._generate_price_intervals(data.min(), data.max())
        self.intervals_number = len(self.intervals)

    def reduction(self):
        used_labels = sorted(set(self.data))
        self.intervals = [self.intervals[i] for i in used_labels]

        old_to_new = {old: new for new, old in enumerate(used_labels)}
        self.data = [old_to_new[label] for label in self.data]

    def classify(self):
        self.data = self._classify(self.data)

    def _classify(self, data: pd.Series) -> pd.Series:
        def find_class(value):
            for idx, (lower, upper) in enumerate(self.intervals):
                if lower <= value <= upper:
                    return idx
            return self.intervals_number - 1

        return data.apply(find_class).astype(int)

    def _generate_price_intervals(self, A: int, B: int) -> List[Tuple[float, float]]:
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
        return self._classify(y_pred)
