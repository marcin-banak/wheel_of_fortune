import json


def load_hyperparameters(hyperparameters_path):
    with open(hyperparameters_path, "r") as f:
        return json.load(f)
