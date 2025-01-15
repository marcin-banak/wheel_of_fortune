from sklearn.utils import resample
import numpy as np
import pandas as pd
from typing import Tuple


def stratified_resample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples_ratio:
    float = 1.0,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.Series]:
    
    numpy_X = X.to_numpy()
    numpy_y = y.to_numpy()
    # Unikalne klasy i liczby próbek
    classes, counts = np.unique(y, return_counts=True)
    
    # Przygotowanie liczby próbek na klasę
    if isinstance(n_samples_ratio, dict):
        n_samples_per_class = [max(n_samples_ratio.get(c, counts[i]), 1) for i, c in enumerate(classes)]
    else:
        n_samples_per_class = [max(int(counts[i] * n_samples_ratio), 1) for i, c in enumerate(classes)]
    
    # Stratyfikowane próbkowanie
    X_resampled = []
    y_resampled = []
    
    for c, n_samples in zip(classes, n_samples_per_class):
        # Wybór próbek dla danej klasy
        X_class = numpy_X[y == c]
        y_class = numpy_y[y == c]
        
        # Próbkowanie z resample
        X_class_resampled, y_class_resampled = resample(
            X_class, y_class,
            n_samples=n_samples,
            random_state=random_state,
            replace=(n_samples > len(y_class))  # Pozwala na próbkowanie z powtórzeniami
        )
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)
    
    # Łączenie wyników
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)
    
    return X_resampled, y_resampled