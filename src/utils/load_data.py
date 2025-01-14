from common.Config import Config
from common.exceptions import NoProcessedDataFile
from preprocessing.dtype_mapping import dtype_mapping
import pandas as pd
from typing import Tuple


def load_data(sample: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    if not Config.processed_data_path.exists():
        raise NoProcessedDataFile(
            f"There is no processed data file in location {Config.processed_data_path}"
            "Run preprocessing.ipynb notebook first!"
        )
    data = pd.read_csv(Config.processed_data_path, low_memory=False)
    if sample:
        data = data.sample(sample)
    data = dtype_mapping()
    for col in data.select_dtypes(include=["category", "object"]).columns:
        data[col] = data[col].astype("category").cat.codes

    X = data.iloc[:, 1:]
    y = data["Price"]

    return X, y