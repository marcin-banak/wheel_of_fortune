from abc import ABC, abstractmethod
from dataclasses import dataclass
from evaluation import AbstractEvaluationResults
import numpy as np


@dataclass
class AbstractHyperparams:
    pass


class AbstractModel(ABC):
    params: AbstractHyperparams

    @abstractmethod
    def eval(self, y_pred: np.ndarray, y_test: np.ndarray) -> AbstractEvaluationResults:
        NotImplementedError

    @abstractmethod
    def save_model(self, filename: str):
        NotImplementedError

    @abstractmethod
    def load_model(self, filename: str):
        NotImplementedError
