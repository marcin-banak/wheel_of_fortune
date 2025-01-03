from typing import List, Tuple
import pandas as pd


def classify(
    data: pd.DataFrame, column: str, intervals: List[Tuple[int, int]]
) -> pd.DataFrame:
    def find_class(value):
        for idx, (lower, upper) in enumerate(intervals):
            if lower <= value <= upper:
                return idx
        return len(intervals) - 1

    return data[column].apply(find_class).astype(int)
