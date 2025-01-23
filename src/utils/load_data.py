from typing import Tuple

import pandas as pd

from src.common.Config import Config
from src.common.exceptions import NoProcessedDataFile
from src.preprocessing.dtype_mapping import dtype_mapping


def load_data(sample: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and processes the dataset.

    :param sample: Optional; Number of rows to sample from the dataset. If 0, loads the entire dataset.
    :return: A tuple containing the feature DataFrame (X) and the target Series (y).
    :raises NoProcessedDataFile: If the processed data file does not exist.
    """
    if not Config.processed_data_path.exists():
        raise NoProcessedDataFile(
            f"There is no processed data file in location {Config.processed_data_path}. "
            "Run preprocessing.ipynb notebook first!"
        )

    # Load the dataset
    data = pd.read_csv(Config.processed_data_path, low_memory=False)

    # Sample the data if required
    if sample:
        data = data.sample(sample)

    # Map data types to specified types
    data = dtype_mapping(data)

    # Encode categorical features as integer codes
    for col in data.select_dtypes(include=["category", "object"]).columns:
        data[col] = data[col].astype("category").cat.codes

    # Separate features and target variable
    X = data.iloc[:, 1:]
    y = data["Price"]

    return X, y
