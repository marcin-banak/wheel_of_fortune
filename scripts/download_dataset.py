import os

from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

TARGET_DIR =  Path(__file__).parent.parent / "data"


def download_kaggle_dataset(dataset):
    api = KaggleApi()
    api.authenticate()

    TARGET_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Downloading {dataset} to {TARGET_DIR}...")
    api.dataset_download_files(dataset, path=TARGET_DIR, unzip=True)
    print("Download complete!")


if __name__ == "__main__":
    dataset_name = "bartoszpieniak/poland-cars-for-sale-dataset"
    download_kaggle_dataset(dataset_name)
