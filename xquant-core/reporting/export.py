import pandas as pd
import json


def export_to_csv(data: pd.DataFrame, filename: str):
    data.to_csv(filename)
    print(f"Data exported to {filename}")


def export_to_json(metrics: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics exported to {filename}")
