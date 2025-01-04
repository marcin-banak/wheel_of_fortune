from itertools import product
from typing import Any, Dict, List


def cartesian_product(param_grid: Dict[str, List[Any]]) -> List[Dict[Any, Any]]:
    """
    Creates cartesian product of parameters.

    Args:
        param_grid (dict): Dictionary containing parameter names and lists of possible values.

    Returns:
        List[Dict]: List of variants of hyperparameters with their names.
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return param_combinations
