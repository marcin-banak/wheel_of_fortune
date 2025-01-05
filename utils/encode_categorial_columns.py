import pandas as pd


def encode_categorical_columns(data):
    for col in data.select_dtypes(include=["category", "object"]).columns:
        data[col] = data[col].astype("category").cat.codes
    return data
