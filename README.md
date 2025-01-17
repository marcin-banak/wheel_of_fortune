# $$\text{wheel of fortune}$$
Machine Learning project that will be used to predict car's value and how it will change in the future.

## Our research task

The research task of this project was to compare how the classification of a continuous numerical set into intervals works compared to the usual use of regression. This can be compared in *comprasion.ipynb* notebook.

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
        - Place the dataset in the **data/** directory and named it "raw_data.csv".

1. Run the main script:
    ```bash
    python main.py
    ```

## How to train main model?

1. Run *preprocessing.ipynb* notebook (you should get *processed_data.csv* in *data* folder)
2. Run *training.ipynb* (you should get *maxed.ubj* in *saved_models* folder)
3. Copy model into your app and load it by using:

```python
xgboost.XGBoostRegressor.load_model("maxed.ubj")
```

## Other notebooks

- *tuning.ipynb* was used for hyperparameter tuning of every model using Bayesian
- *bootstraping.ipynb* was used for bootstraping model training
- *comprasion.ipynb* generates plot which compares different models (Classification vs Regression)
- *model_maxing.ipynb* was used for ultimate hyperparameters optimization of AdvancedRegressionModel
