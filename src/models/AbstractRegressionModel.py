import pandas as pd

from src.evaluation.ClassificationEvaluationResults import ClassificationEvaluationResults
from src.evaluation.RegressionEvaluationResults import RegressionEvaluationResults
from src.models.AbstractModel import AbstractModel
from src.utils.IntervalsHandler import IntervalsHandler


class AbstractRegressionModel(AbstractModel):
    """
    Abstract base class for regression models.

    Inherits from AbstractModel and provides methods for evaluating regression
    and classification predictions.
    """

    def eval(self, y_pred: pd.Series, y_test: pd.Series) -> RegressionEvaluationResults:
        """
        Evaluates regression predictions using ground truth values.

        :param y_pred: Predicted values.
        :param y_test: Ground truth values.
        :return: RegressionEvaluationResults containing evaluation metrics.
        """
        return RegressionEvaluationResults(y_pred, y_test)

    def eval_classification(
        self, y_pred: pd.Series, y_test: pd.Series
    ) -> ClassificationEvaluationResults:
        """
        Evaluates classification predictions using ground truth labels.

        :param y_pred: Predicted labels.
        :param y_test: Ground truth labels.
        :return: ClassificationEvaluationResults containing evaluation metrics.
        """
        return ClassificationEvaluationResults(y_pred, y_test)

    def score_classification(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        interval_handler: IntervalsHandler,
    ) -> ClassificationEvaluationResults:
        """
        Scores the model by converting regression predictions to classification
        labels and evaluating against ground truth labels.

        :param X_test: Test features.
        :param y_test: Ground truth classification labels.
        :param interval_handler: Handler for mapping regression predictions to classification labels.
        :return: ClassificationEvaluationResults containing evaluation metrics.
        """
        return self.eval_classification(
            interval_handler.regression_to_classification(self.predict(X_test)), y_test
        )
