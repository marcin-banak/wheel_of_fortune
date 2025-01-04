from typing import List, Tuple

import pandas as pd


def class_reduction(
    labels: pd.Series, intervals: List[Tuple[int, int]]
) -> Tuple[pd.Series, List[Tuple[int, int]]]:
    used_labels = sorted(set(labels))
    new_intervals = [intervals[i] for i in used_labels]

    old_to_new = {old: new for new, old in enumerate(used_labels)}
    new_labels = [old_to_new[label] for label in labels]
    return new_labels, new_intervals
