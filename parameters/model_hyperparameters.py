from dataclasses import dataclass



@dataclass
class ModelHyperparams:
    pass


@dataclass
class XGBoostHyperparams(ModelHyperparams):
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    max_depth: int
    n_estimators: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float