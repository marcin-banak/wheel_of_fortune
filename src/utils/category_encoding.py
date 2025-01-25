import pandas as pd
import json
from pathlib import Path

def category_encoding(data: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    mapping = {}

    for col in data.select_dtypes(include=["category", "object"]).columns:
        data[col] = data[col].astype("category")
        cat_mapping = dict(enumerate(data[col].cat.categories))
        mapping[col] = cat_mapping
        data[col] = data[col].cat.codes

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    return data
