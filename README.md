# $$\text{wheel of fortune}$$
Machine Learning project that will be used to predict car's value and how it will change in the future.

## Steps to set up the project

1. Clone the [repository](https://github.com/marcin-banak/wheel_of_fortune.git)
    ```
    git clone https://github.com/marcin-banak/wheel_of_fortune.git
    ```

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

1. Download the dataset:
    - Automatically:
        1.  Set up Kaggle API credentials:

            - Download ***kaggle.json*** from Kaggle **Account Settings**.
            - Place ***kaggle.json*** in the .kaggle/ folder in your home directory:
                - Linux/Mac: **~/.kaggle/kaggle.json**
                - Windows: **%USERPROFILE%\.kaggle\kaggle.json**
        1.  Run the script to download the dataset:
            ```bash
            python download_dataset.py
            ```
            The dataset will be downloaded and extracted into the data/ directory.

    - Manually:
        - Go to [dataset on kaggle](https://www.kaggle.com/datasets/bartoszpieniak/poland-cars-for-sale-dataset) and download it.
        - Place the dataset in the **data/** directory.

1. Run the main script:
    ```bash
    python main.py
    ```