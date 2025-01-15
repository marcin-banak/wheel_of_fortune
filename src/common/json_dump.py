import numpy as np
from pathlib import Path
from src.common.types import Params
import json


def json_dump(obj: Params, path: Path):
    def numpy_to_python(obj):
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
