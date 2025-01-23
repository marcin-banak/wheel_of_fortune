import json
from pathlib import Path

import numpy as np

from src.common.types import Params


def json_dump(obj: Params, path: Path):
    """
    Serializes an object to a JSON file, handling NumPy data types.

    :param obj: The object to serialize.
    :param path: The file path where the JSON file will be saved.
    """

    def numpy_to_python(obj):
        """
        Converts NumPy data types to native Python data types for JSON serialization.

        :param obj: The object to convert.
        :return: The converted object.
        :raises TypeError: If the object is not JSON serializable.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.isscalar(obj):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=numpy_to_python)
