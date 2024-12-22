import json
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi import HTTPException
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from google.cloud import firestore

IRIS_DATASET_PATH = "src/data/iris/iris.csv"
CONFIG_PATH = "C:\\Users\\C Rémy\\OneDrive - Fondation EPF\\Documents\\klemLeti\\API---Webscrapping\\TP2 and  3\\services\\epf-flower-data-science\\src\\config\\datasets.json"
DATASET_PATH = "src/data"
MODEL_PARAMETERS_PATH = "src/config/model_parameters.json"
MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google.json"

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

    import pandas as pd
    df = pd.DataFrame(data_json)

    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    X = df.drop(columns=[target_column])
    y = df[target_column]

    params = load_model_parameters(model_name)

    if model_name == "LogisticRegression":
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    return model_path



def load_trained_model(model_name: str):
    """
    Load the trained model from the models directory.
    
    Parameters:
        model_name (str): Name of the model to load (e.g., 'LogisticRegression').

    Returns:
        model: The trained model.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found at {model_path}.")
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {e}")


def make_predictions(model, data):
    """
    Make predictions using the trained model.

    Parameters:
        model: The trained classification model.
        input_data (List[Dict]): Input features in JSON format.

    Returns:
        List: Predictions made by the model.
    """
    try:
        # Convert input JSON data to a pandas DataFrame
        input_df = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(input_df)

        # Return predictions as a list
        return predictions.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {e}")
    
def get_firestore_parameters():
    """
    Retrieve parameters from Firestore.
    
    Returns:
        dict: The parameters stored in the Firestore 'parameters' collection.
    """
    try:
        db = firestore.Client()
        doc_ref = db.collection("parameters").document("parameters")
        doc = doc_ref.get()

        if not doc.exists:
            raise ValueError("The document 'parameters' does not exist in Firestore.")

        return doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve parameters: {e}")

def add_or_update_parameters(data: dict):
    """
    Add or update parameters in Firestore.

    Args:
        data (dict): A dictionary of parameters to add or update.

    Returns:
        dict: Confirmation message with updated data.
    """
    try:
        db = firestore.Client()

        doc_ref = db.collection("parameters").document("parameters")

        doc_ref.set(data, merge=True)

        return {"message": "Parameters updated successfully.", "updated_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update parameters: {e}")