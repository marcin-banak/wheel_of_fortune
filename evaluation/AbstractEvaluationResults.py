from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractEvaluationResults(ABC):
    @abstractmethod
    def get_score(self):
        pass

    @property
    def ideal_distance(self):
        pass
