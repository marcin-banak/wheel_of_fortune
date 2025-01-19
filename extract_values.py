# Script to structurally extract unique values from dataset for use in project web application
import json
from dataclasses import dataclass
from typing import List
import pandas as pd


PROCESSED_DATA_PATH = "data/processed_car_sale_ads.csv"
RAW_DATA_PATH = "data/raw_data.csv"

# Read datasets
processed_data = pd.read_csv(PROCESSED_DATA_PATH)
raw_data = pd.read_csv(RAW_DATA_PATH)

# Handle NaNs
processed_data = processed_data.fillna({'Vehicle_brand': '', 'Vehicle_model': '', 'Vehicle_generation': ''})
raw_data = raw_data.fillna({'Vehicle_model': '', 'Vehicle_version': '', 'Features': '[]'})


@dataclass
class VehicleModel:
    vehicle_model: str
    vehicle_version: List[str]
    vehicle_generation: List[str]


@dataclass
class VehicleBrand:
    name: str
    vehicle_model: List[VehicleModel]

@dataclass
class ResultJSON:
    condition: List[str]
    vehicle_brand: List[VehicleBrand]
    fuel_type: List[str]
    drive: List[str]
    transmission: List[str]
    type: List[str]
    doors_number: List[int]
    colour: List[str]
    features: List[str]


def brand_information() -> List[VehicleBrand]:
    # All brands
    brands = processed_data['Vehicle_brand'].unique().tolist()

    # All models for brand
    brand_model_dict = processed_data.groupby('Vehicle_brand')['Vehicle_model'].unique().to_dict()


    # All generations for model
    model_generation_dict = processed_data.groupby('Vehicle_model')['Vehicle_generation'].unique().apply(list).to_dict()

    # All versions for model
    model_version_dict = raw_data.groupby('Vehicle_model')['Vehicle_version'].unique().apply(list).to_dict()

    result = []

    for brand in brands:
        vehicle_models = []
        for model in brand_model_dict[brand]:
            vehicle_models.append(VehicleModel(
                vehicle_model=model,
                vehicle_version=model_version_dict.get(model, []),
                vehicle_generation=model_generation_dict.get(model, [])
            ))
        result.append(VehicleBrand(
            name=brand,
            vehicle_model=vehicle_models
        ))

    return result

def dataset_values() -> ResultJSON:
    condition = processed_data['Condition'].unique().tolist()
    fuel_type = processed_data['Fuel_type'].unique().tolist()
    drive = processed_data['Drive'].unique().tolist()
    transmission = processed_data['Transmission'].unique().tolist()
    type_ = processed_data['Type'].unique().tolist()
    doors_number = processed_data['Doors_number'].unique().tolist()
    colour = processed_data['Colour'].unique().tolist()
    features = list(set([feature for sublist in raw_data['Features'].dropna().apply(eval) for feature in sublist]))

    result = ResultJSON(
        condition=condition,
        vehicle_brand=brand_information(),
        fuel_type=fuel_type,
        drive=drive,
        transmission=transmission,
        type=type_,
        doors_number=doors_number,
        colour=colour,
        features=features)

    return result

def export_to_json(data: ResultJSON, file_path: str):
    with open(file_path, 'w') as json_file:
        json.dump(data.__dict__, json_file, default=lambda o: o.__dict__, indent=4)



data = dataset_values()
export_to_json(data, "data/unique_values.json")