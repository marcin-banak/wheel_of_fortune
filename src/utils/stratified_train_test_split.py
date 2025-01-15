import numpy as np
import pandas as pd


def stratified_train_test_split(X, y, test_size=0.3, random_state=None):
    np.random.seed(random_state)
    y = np.array(y)

    # Ensure at least one sample of each class in the training set
    unique_classes, indices = np.unique(y, return_index=True)
    X_initial_train = X.iloc[indices]
    y_initial_train = y[indices]

    # Remove these samples from the dataset
    mask = np.ones(len(X), dtype=bool)
    mask[indices] = False
    X_remaining = X[mask]
    y_remaining = y[mask]

    # Split remaining data into training and test sets
    train_indices = []
    test_indices = []

    # Iterate over each class
    for cls in np.unique(y_remaining):
        cls_indices = np.where(y_remaining == cls)[0]
        if len(cls_indices) > 1:
            n_test = max(1, int(test_size * len(cls_indices)))
            test_cls_indices = np.random.choice(cls_indices, size=n_test, replace=False)
            train_cls_indices = list(set(cls_indices) - set(test_cls_indices))
            test_indices.extend(test_cls_indices)
            train_indices.extend(train_cls_indices)
        else:
            train_indices.extend(
                cls_indices
            )

    X_train = pd.concat([X_initial_train, X_remaining.iloc[train_indices]])
    y_train = list(y_initial_train) + list(y_remaining[train_indices])

    X_test = X_remaining.iloc[test_indices]
    y_test = list(y_remaining[test_indices])

    return X_train, X_test, y_train, y_test
