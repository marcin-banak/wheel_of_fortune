from dataclasses import fields
from skopt.space import Categorical, Integer, Real, Dimension
from models.AbstractModel import AbstractHyperparams
from typing import List


def create_hyperparams_space(hyperparams_class: AbstractHyperparams) -> List[Dimension]:
    dimensions = []

    for field in fields(hyperparams_class):
        if not isinstance(field.metadata.get("space"), tuple):
            raise ValueError(
                f"Field '{field.name}' must define 'space' metadata as a tuple."
            )

        param_range = field.metadata["space"]
        param_type = field.metadata.get("type", "float")

        if param_type == "float":
            dimensions.append(Real(*param_range, name=field.name))
        elif param_type == "int":
            dimensions.append(Integer(*param_range, name=field.name))
        elif param_type == "categorical":
            dimensions.append(Categorical(param_range, name=field.name))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

    return dimensions