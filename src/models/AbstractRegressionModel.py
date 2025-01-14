from models.AbstractModel import AbstractModel
from evaluation.RegressionEvaluationResults import RegressionEvaluationResults
from evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults
from common.IntervalsHandler import IntervalsHandler
from utils.classify import classify
import pandas as pd


class AbstractRegressionModel(AbstractModel):
    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> RegressionEvaluationResults:
        return RegressionEvaluationResults(y_pred, y_test)
    
    def eval_classification(
        self, y_pred: pd.Series, y_test: pd.Series, intervals_handler: IntervalsHandler
    ) -> ClassificationEvaluationResults:
        y_pred_class = classify(pd.Series(y_pred), intervals_handler.intervals)
        y_test_class = classify(pd.Series(y_test), intervals_handler.intervals)
        return ClassificationEvaluationResults(y_pred_class, y_test_class)
