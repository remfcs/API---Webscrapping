import json
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi import HTTPException
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

IRIS_DATASET_PATH = "src/data/iris/iris.csv"
CONFIG_PATH = "src/config/datasets.json"
DATASET_PATH = "src/data"

def save_config(config):
    """
    Save updates to the JSON configuration file.
    """
    try:
        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config file: {e}")


def load_config():
    """
    Load the JSON configuration file.
    If the file does not exist or is invalid, raise an appropriate HTTP exception.
    """
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=503, detail="Config file not found.")
    try:
        with open(CONFIG_PATH, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format in config file.")

import opendatasets as od   
    
def download_dataset_from_url(dataset_name: str, url: str):
    """
    Download a dataset from Kaggle using OpenDataset, parse it as a DataFrame, and return it as JSON.
    """
    os.makedirs(DATASET_PATH, exist_ok=True)  # Ensure the data directory exists
    csv_path = os.path.join(DATASET_PATH, f"{dataset_name}.csv")

    # Download the dataset using OpenDataset
    try:
        od.download_kaggle_dataset(url, DATASET_PATH)
        print(f"Dataset '{dataset_name}' downloaded and saved to: {DATASET_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset '{dataset_name}': {e}")

    # Parse and return the dataset as JSON
    try:
        df = pd.read_csv(IRIS_DATASET_PATH)

        # Handle NaN, inf, -inf for JSON compliance
        df = df.replace([float('inf'), float('-inf')], None)
        df = df.where(pd.notnull(df), None)
        print("check3")

        # Convert DataFrame to JSON
        json_data = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse dataset '{dataset_name}' to JSON: {str(e)}")


def process_dataset(csv_path: str):
    """
    Perform necessary processing on the entire dataset and return it as a JSON.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        df['Species'] = df['Species'].str.split('-').str[1]
        json_data = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {e}")
