from pathlib import Path


class Config:

    # paths
    root_dir = Path(__file__).parent.parent.parent
    saved_models_dir = root_dir / "saved_models"
    trained_models_dir = root_dir / "trained_models"
    data_path = root_dir / "data" / "raw_data.csv"
    processed_data_path = root_dir / "data" / "processed_data.csv"
    category_mapping_path = root_dir / "data" / "category_mapping.json"
    future_prediction_processed_data_path = (
        root_dir / "data" / "future_prediction_processed_data.csv"
    )
    saved_hyperparameters_dir = root_dir / "saved_hyperparameters"
    optimized_hyperparameters_dir = root_dir / "optimized_hyperparameters"

    # preprocessing
    actual_year = 2025
