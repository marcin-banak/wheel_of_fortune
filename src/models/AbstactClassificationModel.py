import pandas as pd

from src.evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults
from src.models.AbstractModel import AbstractModel


class AbstractClassificationModel(AbstractModel):
    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> ClassificationEvaluationResults:
        return ClassificationEvaluationResults(y_pred, y_test)
