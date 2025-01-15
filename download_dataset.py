import os

from kaggle.api.kaggle_api_extended import KaggleApi

from src.common.Config import Config


def download_kaggle_dataset(dataset):
    api = KaggleApi()
    api.authenticate()

    Config.data_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"Downloading {dataset} to {Config.data_path}...")
    api.dataset_download_files(dataset, path=Config.data_path.parent, unzip=True)
    os.rename(Config.data_path.parent / "Car_sale_ads.csv", Config.data_path)
    print("Download complete!")


if __name__ == "__main__":
    dataset_name = "bartoszpieniak/poland-cars-for-sale-dataset"
    download_kaggle_dataset(dataset_name)
