import json
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi import HTTPException
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression

IRIS_DATASET_PATH = "src/data/iris/iris.csv"
CONFIG_PATH = "src/config/datasets.json"
DATASET_PATH = "src/data"
MODEL_PARAMETERS_PATH = "src/config/model_parameters.json"
MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)
    
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
        print("y")
        df = pd.read_csv(csv_path)
        print("z")
        df['Species'] = df['Species'].str.split('-').str[1]
        json_data = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {e}")


def split_set(data_json, test_size=0.2, random_state=42):
    """
    Split a processed dataset (provided as JSON) into train and test sets.

    Parameters:
        data_json (list or dict): The dataset to split, provided as a JSON-like structure (list of dictionaries).
        test_size (float): Fraction of the dataset to use as the test set. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        dict: A dictionary containing the train and test datasets as JSON.
    """
    try:
        # Convert JSON input to DataFrame
        df = pd.DataFrame(data_json)

        # Split the dataset
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        train_json = train_df.to_dict(orient="records")
        test_json = test_df.to_dict(orient="records")

        return {
            "message": "Dataset split successfully.",
            "train_data": train_json,
            "test_data": test_json,
        }
    except Exception as e:
        raise ValueError(f"Failed to split dataset: {e}")
    
def load_model_parameters(model_name: str):
    """
    Load model parameters from the configuration file.
    """
    if not os.path.exists(MODEL_PARAMETERS_PATH):
        raise FileNotFoundError(f"Configuration file not found at {MODEL_PARAMETERS_PATH}.")
    
    with open(MODEL_PARAMETERS_PATH, "r") as file:
        config = json.load(file)
    
    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not found in configuration file.")
    
    return config[model_name]

def train_model(data_json, model_name="LogisticRegression", target_column="Species"):
    """
    Train a classification model and save it to the models directory.

    Parameters:
        data_json (list): The processed dataset in JSON format.
        model_name (str): The name of the model to train.
        target_column (str): The column to use as the target variable.

    Returns:
        str: Path to the saved model.
    """
    from sklearn.model_selection import train_test_split

    # Convert JSON to DataFrame
    import pandas as pd
    df = pd.DataFrame(data_json)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Load model parameters
    params = load_model_parameters(model_name)

    # Train the model
    if model_name == "LogisticRegression":
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    return model_path