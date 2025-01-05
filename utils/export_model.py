import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Type

from models.AbstractModel import AbstractHyperparams, AbstractModel

EXPORT_DIR = Path(__file__).parent.parent / "trained_models"


def save_model(model: AbstractModel, filename: str):
    """Export model into pickle file

    :param model: Model to export
    :param filename: Name of model file
    """

    with open(EXPORT_DIR / f"{filename}.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str) -> AbstractModel:
    """Loads model from pickle file

    :param filename: Name of model to load
    :return: Loaded model
    """

    with open(EXPORT_DIR / f"{filename}.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model
