from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """
    Base class for evaluation results.
    """

    pass


@dataclass
class ClassificationEvaluationResults(EvaluationResults):
    """
    Class to store and evaluate classification results.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float

    def cost_function(self) -> float:
        """
        Computes a weighted cost function for classification metrics.
        Higher scores indicate a better model.

        :returns: The weighted sum of classification metrics.
        """
        return self.f1 * 0.4 + self.accuracy * 0.3 + self.precision * 0.2 + self.recall * 0.1

    def __gt__(self, other) -> bool:
        """
        Compares two ClassificationEvaluationResults objects using the cost function.

        other: Another ClassificationEvaluationResults object to compare with.
        :returns: True if the current object has a higher cost function score, False otherwise.
        """
        if not isinstance(other, ClassificationEvaluationResults):
            raise TypeError(
                "Comparison is only supported between ClassificationEvaluationResults objects."
            )

        return self.cost_function() > other.cost_function()


@dataclass
class RegressionEvaluationResults(EvaluationResults):
    """
    Class to store and evaluate regression results.
    """

    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float

    def cost_function(self) -> float:
        """
        Computes a weighted cost function for regression metrics.
        Lower scores indicate a better model.

        :returns: The weighted sum of regression metrics.
        """
        return self.mae * 0.4 + self.rmse * 0.3 + self.mape * 0.2 + self.r2 * -0.2

    def __gt__(self, other) -> bool:
        """
        Compares two RegressionEvaluationResults objects using the cost function.

        other : Another RegressionEvaluationResults object to compare with.
        :returns: True if the current object has a lower cost function score, False otherwise.
        """
        if not isinstance(other, RegressionEvaluationResults):
            raise TypeError(
                "Comparison is only supported between RegressionEvaluationResults objects."
            )
        return self.cost_function() < other.cost_function()
