from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample


def stratified_resample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples_ratio: float = 1.0,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Performs stratified resampling of the dataset, maintaining class proportions.

    :param X: Feature DataFrame.
    :param y: Target Series.
    :param n_samples_ratio: Ratio of samples to generate per class. Can be a float or a dictionary
                            mapping class labels to sample counts.
    :param random_state: Random seed for reproducibility.
    :return: A tuple of resampled feature matrix (X) and target array (y).
    """
    numpy_X = X.to_numpy()
    numpy_y = y.to_numpy()

    # Unique classes and their sample counts
    classes, counts = np.unique(y, return_counts=True)

    # Calculate the number of samples per class
    if isinstance(n_samples_ratio, dict):
        n_samples_per_class = [
            max(n_samples_ratio.get(c, counts[i]), 1) for i, c in enumerate(classes)
        ]
    else:
        n_samples_per_class = [
            max(int(counts[i] * n_samples_ratio), 1) for i, c in enumerate(classes)
        ]

    # Stratified resampling
    X_resampled = []
    y_resampled = []

    for c, n_samples in zip(classes, n_samples_per_class):
        # Select samples for the current class
        X_class = numpy_X[y == c]
        y_class = numpy_y[y == c]

        # Resample with or without replacement
        X_class_resampled, y_class_resampled = resample(
            X_class,
            y_class,
            n_samples=n_samples,
            random_state=random_state,
            replace=(
                n_samples > len(y_class)
            ),  # Allow sampling with replacement if needed
        )
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)

    # Combine results
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    return X_resampled, y_resampled
