import pandas as pd


def dtype_mapping(data: pd.DataFrame) -> pd.DataFrame:
    dtype_mapping = {
        "Price": "Int64",
        "Condition": "category",
        "Vehicle_brand": "category",
        "Vehicle_model": "category",
        "Production_year": "Int64",
        "Mileage_km": "Int64",
        "Power_HP": "Int64",
        "Displacement_cm3": "Int64",
        "Fuel_type": "category",
        "Drive": "category",
        "Transmission": "category",
        "Type": "category",
        "Doors_number": "Int64",
        "Colour": "category",
        "First_owner": "boolean",
        "Vehicle_generation": "category",
        "Advanced_model": "category",
    }

    return data.astype(dtype_mapping)
