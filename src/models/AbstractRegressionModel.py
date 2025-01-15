import pandas as pd

from src.evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults
from src.evaluation.RegressionEvaluationResults import RegressionEvaluationResults
from src.models.AbstractModel import AbstractModel
from src.utils.IntervalsHandler import IntervalsHandler


class AbstractRegressionModel(AbstractModel):
    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> RegressionEvaluationResults:
        return RegressionEvaluationResults(y_pred, y_test)
    
    def eval_classification(
        self, y_pred: pd.Series, y_test: pd.Series
    ) -> ClassificationEvaluationResults:
        return ClassificationEvaluationResults(y_pred, y_test)
    
    def score_classification(
        self, X_test: pd.DataFrame, y_test: pd.Series, interval_handler: IntervalsHandler
    ) -> ClassificationEvaluationResults:
        return self.eval_classification(
            interval_handler.regression_to_classification(self.predict(X_test)),
            y_test
        )
