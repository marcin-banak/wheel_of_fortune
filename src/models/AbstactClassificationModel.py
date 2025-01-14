from models.AbstractModel import AbstractModel
import pandas as pd
from evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults


class AbstractClassificationModel(AbstractModel):
    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> ClassificationEvaluationResults:
        return ClassificationEvaluationResults(y_pred, y_test)
