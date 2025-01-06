import numpy as np


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
