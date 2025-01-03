import os

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset, target_dir="../data"):
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Downloading {dataset} to {target_dir}...")
    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    print("Download complete!")


if __name__ == "__main__":
    dataset_name = "bartoszpieniak/poland-cars-for-sale-dataset"
    download_kaggle_dataset(dataset_name)
