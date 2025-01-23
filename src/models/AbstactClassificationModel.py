import pandas as pd

from src.evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults
from src.models.AbstractModel import AbstractModel


class AbstractClassificationModel(AbstractModel):
    """
    Abstract base class for classification models.

    Inherits from AbstractModel and provides a method to evaluate classification results.
    """

    def eval(
        self, y_pred: pd.Series, y_test: pd.Series
    ) -> ClassificationEvaluationResults:
        """
        Evaluates classification predictions using ground truth labels.

        :param y_pred: Predicted labels.
        :param y_test: Ground truth labels.
        :return: ClassificationEvaluationResults containing evaluation metrics.
        """
        return ClassificationEvaluationResults(y_pred, y_test)
