import pandas as pd


def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y