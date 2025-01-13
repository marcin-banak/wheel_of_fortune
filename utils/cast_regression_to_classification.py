from utils.export_model import load_model
import pandas as pd
from data_processing.dtype_mapping import dtype_mapping
from sklearn.model_selection import train_test_split
from utils.classification_range_generator import generate_price_intervals
from utils.classify import classify
from utils.class_reduction import class_reduction
from evaluation.evaluate_classification import evaluate_classification
from evaluation.AbstractEvaluationResults import MetricEnum
from utils.encode_categorial_columns import encode_categorical_columns
import numpy as np


def cast_regression_to_classification(
    model_path: str,
    interval_func,
    metric: MetricEnum = None,
    category_encoding: bool = False,
):
    model = load_model(model_path)

    data = pd.read_csv("../data/processed_car_sale_ads.csv", low_memory=False)
    data = dtype_mapping(data)

    if category_encoding:
        encode_categorical_columns(data)

    X = data.iloc[:, 1:]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # model.fit(X_train, y_train)

    y_pred_regression = model.predict(X_test)

    intervals = generate_price_intervals(y.min(), y.max(), interval_func)

    y_pred_class = classify(pd.Series(y_pred_regression), intervals)
    y_test_class = classify(pd.Series(y_test), intervals)

    # y_test_class, intervals = class_reduction(pd.Series(y_test_class), intervals)
    # y_pred_class, _ = class_reduction(pd.Series(y_pred_class), intervals)

    metrics = evaluate_classification(np.array(y_pred_class), np.array(y_test_class))

    print("Classification metrics:\n", metrics)

    if metric:
        try:
            return metrics.get_metric(metric), intervals
        except:
            print(f"Metric {metric} is not common for classificalion")

    return model, intervals
