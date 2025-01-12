import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)


from dataclasses import dataclass

from utils.export_model import load_model
import pandas as pd
from data_processing.dtype_mapping import dtype_mapping
from sklearn.model_selection import train_test_split
from utils.classification_range_generator import generate_price_intervals
from utils.classify import classify

@dataclass
class ConsistencyEvaluationResults:
    consistency_percentage: float
    average_distance: float

def evaluate_model_consistency(
    regression_model_path: str,
    classification_model_path: str,
    interval_func,
    category_encoding: bool = False
) -> ConsistencyEvaluationResults:
    """
        Checks the consistency of regression model predictions with classifier intervals.

        Parameters:
            regression_model_path: Path to the regression model.
            classification_model_path: Path to the classification model.
            interval_func: Function to generate price intervals.

        Returns:
            ConsistencyEvaluationResults: An object containing the percentage of consistency 
            and the average distance from the interval.
    """
    regression_model = load_model(regression_model_path)
    claddification_model = load_model(classification_model_path)

    data = pd.read_csv("../data/processed_car_sale_ads.csv", low_memory=False)
    data = dtype_mapping(data)

    X = data.iloc[:, 1:]
    y = data["Price"]

    if category_encoding:
        X = pd.get_dummies(X, drop_first=True)  

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    price_intervals = generate_price_intervals(y.min(), y.max(), interval_func)

    y_train_classified = classify(y_train, price_intervals)
    y_test_classified = classify(y_test, price_intervals)

    regression_model.fit(X_train, y_train)
    claddification_model.fit(X_train, y_train_classified)

    regression_predictions = regression_model.predict(X_test)
    classifier_labels = claddification_model.predict(X_test)

    classifier_intervals = [price_intervals[label] for label in classifier_labels]

    total_cases = len(regression_predictions)
    consistent_count = 0
    total_distance = 0

    for prediction, (lower, upper) in zip(regression_predictions, classifier_intervals):
        if lower <= prediction <= upper:
            consistent_count += 1
        else:
            # Obliczanie odległości od najbliższego końca przedziału
            if prediction < lower:
                total_distance += lower - prediction
            elif prediction > upper:
                total_distance += prediction - upper

    consistency_percentage = (consistent_count / total_cases) * 100

    inconsistent_cases = total_cases - consistent_count
    average_distance = (
        total_distance / inconsistent_cases if inconsistent_cases > 0 else 0
    )

    return ConsistencyEvaluationResults(
        consistency_percentage=consistency_percentage, 
        average_distance=average_distance
    )