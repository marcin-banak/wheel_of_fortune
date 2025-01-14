from typing import List, Tuple

import pandas as pd


def classify(data: pd.Series, intervals: List[Tuple[int, int]]) -> pd.Series:
    def find_class(value):
        for idx, (lower, upper) in enumerate(intervals):
            if lower <= value <= upper:
                return idx
        return len(intervals) - 1

    return data.apply(find_class).astype(int)
