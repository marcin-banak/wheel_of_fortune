from pprint import pprint

import numpy as np

# Original feature weights from training results
train_result_weights = {
     "Condition": 690.0,
    "Vehicle_brand": 12182.0,
    "Vehicle_model": 16335.0,
    "Vehicle_generation": 12095.0,
    "Production_year": 14682.0,
    "Mileage_km": 18262.0,
    "Power_HP": 13732.0,
    "Displacement_cm3": 9310.0,
    "Fuel_type": 3321.0,
    "Drive": 2756.0,
    "Transmission": 1516.0,
    "Type": 7042.0,
    "Doors_number": 2027.0,
    "Colour": 11129.0,
    "First_owner": 2095.0,
    "Advanced_model": 6613.0,
    "ABS": 331.0,
    "ASR (traction control)": 778.0,
    "AUX socket": 850.0,
    "Active cruise control": 559.0,
    "Adjustable suspension": 378.0,
    "Aftermarket radio": 315.0,
    "Air curtains": 771.0,
    "Airbag protecting the knees": 588.0,
    "Alarm": 871.0,
    "Alloy wheels": 1168.0,
    "Automatic air conditioning": 822.0,
    "Auxiliary heating": 289.0,
    "Blind spot sensor": 490.0,
    "Bluetooth": 820.0,
    "CD": 1251.0,
    "CD changer": 367.0,
    "Central locking": 329.0,
    "Cruise control": 790.0,
    "DVD player": 496.0,
    "Daytime running lights": 907.0,
    "Drivers airbag": 397.0,
    "Dual zone air conditioning": 866.0,
    "ESP(stabilization of the track)": 736.0,
    "Electric front windows": 395.0,
    "Electric rear windows": 876.0,
    "Electrically adjustable mirrors": 500.0,
    "Electrically adjustable seats": 587.0,
    "Electrochromic rear view mirror": 640.0,
    "Electrochromic side mirrors": 673.0,
    "Factory radio": 599.0,
    "Fog lights": 812.0,
    "Four-zone air conditioning": 288.0,
    "Front parking sensors": 1017.0,
    "Front side airbags": 817.0,
    "GPS navigation": 923.0,
    "HUD(head-up display)": 249.0,
    "Heated front seats": 1108.0,
    "Heated rear seats": 293.0,
    "Heated side mirrors": 863.0,
    "Heated windscreen": 531.0,
    "Hook": 600.0,
    "Immobilizer": 570.0,
    "Isofix": 855.0,
    "LED lights": 907.0,
    "Lane assistant": 524.0,
    "Leather upholstery": 925.0,
    "MP3": 817.0,
    "Manual air conditioning": 920.0,
    "Multifunction steering wheel": 723.0,
    "On-board computer": 637.0,
    "Panoramic roof": 432.0,
    "Parking assistant": 501.0,
    "Passengers airbag": 415.0,
    "Power steering": 432.0,
    "Rain sensor": 689.0,
    "Rear parking sensors": 876.0,
    "Rear side airbags": 682.0,
    "Rear view camera": 844.0,
    "Roof rails": 693.0,
    "SD socket": 669.0,
    "Shift paddles": 542.0,
    "Speed limiter": 652.0,
    "Start-Stop system": 715.0,
    "Sunroof": 482.0,
    "TV tuner": 108.0,
    "Tinted windows": 1018.0,
    "Twilight sensor": 694.0,
    "USB socket": 796.0,
    "Velor upholstery": 975.0,
    "Xenon lights": 913.0,
}

# List of features to extract weights for
features = [
    "ABS",
    "ASR (traction control)",
    "AUX socket",
    "Active cruise control",
    "Adjustable suspension",
    "Aftermarket radio",
    "Air curtains",
    "Airbag protecting the knees",
    "Alarm",
    "Alloy wheels",
    "Automatic air conditioning",
    "Auxiliary heating",
    "Blind spot sensor",
    "Bluetooth",
    "CD",
    "CD changer",
    "Central locking",
    "Cruise control",
    "DVD player",
    "Daytime running lights",
    "Drivers airbag",
    "Dual zone air conditioning",
    "ESP(stabilization of the track)",
    "Electric front windows",
    "Electric rear windows",
    "Electrically adjustable mirrors",
    "Electrically adjustable seats",
    "Electrochromic rear view mirror",
    "Electrochromic side mirrors",
    "Factory radio",
    "Fog lights",
    "Four-zone air conditioning",
    "Front parking sensors",
    "Front side airbags",
    "GPS navigation",
    "HUD(head-up display)",
    "Heated front seats",
    "Heated rear seats",
    "Heated side mirrors",
    "Heated windscreen",
    "Hook",
    "Immobilizer",
    "Isofix",
    "LED lights",
    "Lane assistant",
    "Leather upholstery",
    "MP3",
    "Manual air conditioning",
    "Multifunction steering wheel",
    "On-board computer",
    "Panoramic roof",
    "Parking assistant",
    "Passengers airbag",
    "Power steering",
    "Rain sensor",
    "Rear parking sensors",
    "Rear side airbags",
    "Rear view camera",
    "Roof rails",
    "SD socket",
    "Shift paddles",
    "Speed limiter",
    "Start-Stop system",
    "Sunroof",
    "TV tuner",
    "Tinted windows",
    "Twilight sensor",
    "USB socket",
    "Velor upholstery",
    "Xenon lights",
]

# Extract weights for the specified features
feature_weights_train_result = {
    feature: train_result_weights[feature] for feature in features
}


# Function to normalize weights to a specified scale
def normalize_weights(weights, scale=100):
    """
    Normalize feature weights to a specified scale.

    :param weights: Dictionary of feature weights.
    :param scale: Target scale for normalization (default is 0-100).
    :return: Dictionary of normalized weights.
    """
    values = np.array(list(weights.values()))
    min_val = values.min()
    max_val = values.max()

    normalized = {
        key: int((value - min_val) / (max_val - min_val) * scale)
        for key, value in weights.items()
    }
    return normalized


# Normalize the extracted feature weights
normalized_weights = normalize_weights(feature_weights_train_result, scale=100)

# Print the normalized weights
pprint(normalized_weights)

# Assign normalized weights to the final feature_weights variable
feature_weights = normalized_weights
