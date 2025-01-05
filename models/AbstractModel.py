from abc import ABC, abstractmethod
from dataclasses import dataclass
from evaluation import AbstractEvaluationResults
import numpy as np
from typing import List, Tuple
import plotly.graph_objects as go


@dataclass
class AbstractHyperparams:
    pass


class AbstractModel(ABC):
    hyperparams: AbstractHyperparams

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> AbstractEvaluationResults:
        return self.eval(self.predict(X_test), y_test)

    def feature_importance(self):
        parameters, values = zip(*sorted(self._feature_importance(), key=lambda x: x[1]))
        fig = go.Figure(data=go.Bar(x=values, y=parameters, orientation="h"))
        fig.update_layout(
            xaxis_title="Weight",
            yaxis_title="Feature Name",
        )
        fig.show()

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        NotImplementedError

    @abstractmethod
    def eval(self, y_pred: np.ndarray, y_test: np.ndarray) -> AbstractEvaluationResults:
        NotImplementedError

    @abstractmethod
    def _feature_importance(self) -> List[Tuple[str, float]]:
        NotImplementedError
