import os
from fastapi import APIRouter, HTTPException
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from src.services.data import *
from typing import List, Dict

router = APIRouter()

# Config paths
IRIS_DATASET_PATH = "src/data/iris/iris.csv"

# Route: Get dataset details
@router.get("/dataset/{dataset_name}", tags=["Dataset"])
async def get_dataset(dataset_name: str):
    """
    Get dataset details by name.
    """
    config = load_config()
    if dataset_name not in config:
        raise HTTPException(status_code=404, detail="Dataset not found in config.")
    return config[dataset_name]

# Route: Add a new dataset
@router.post("/dataset", tags=["Dataset"])
async def add_dataset(name: str, url: str):
    """
    Add a new dataset to the config file.
    """
    config = load_config()
    if name in config:
        raise HTTPException(status_code=400, detail="Dataset already exists.")
    
    config[name] = {"name": name, "url": url}
    save_config(config)
    return {"message": "Dataset added successfully.", "dataset": config[name]}

# Route: Modify an existing dataset
@router.put("/dataset/{dataset_name}", tags=["Dataset"])
async def modify_dataset(dataset_name: str, url: str = None):
    config = load_config()
    if dataset_name not in config:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    if url:
        config[dataset_name]["url"] = url
    
    save_config(config)
    return {"message": "Dataset modified successfully.", "dataset": config[dataset_name]}

# Route: Load dataset and return as JSON
@router.get("/load-dataset/{dataset_name}", tags=["Dataset"])
async def load_dataset(dataset_name: str):
    """
    Load a dataset by its Kaggle name and return it as JSON.
    """
    config = load_config()
    # Check if the dataset exists in the config file
    if dataset_name not in config:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in config.")

    dataset_info = config[dataset_name]
    url = dataset_info.get("url")
    if not url:
        raise HTTPException(status_code=500, detail=f"URL not found for dataset '{dataset_name}'.")
    
    json_file_path = download_dataset_from_url(dataset_name, url)
    return {
        "message": f"Dataset '{dataset_name}' downloaded and converted to JSON successfully.",
        "json_path": json_file_path
    }
    
@router.get("/process-dataset", tags=["Dataset"])
async def process_dataset_endpoint():
    """
    Endpoint to process the dataset.
    """
    if not os.path.exists(IRIS_DATASET_PATH):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {IRIS_DATASET_PATH}")

    # Process the dataset
    result = process_dataset(IRIS_DATASET_PATH)
    return result

@router.get("/split-dataset", tags=["Dataset"])
async def split_dataset_endpoint(test_size: float = 0.2, random_state: int = 42):
    """
    Endpoint to split the processed dataset into train and test sets.
    """
    if not os.path.exists(IRIS_DATASET_PATH):
        raise HTTPException(status_code=404, detail=f"Dataset not found at {DATASET_PATH}")

    # Process the dataset
    df = process_dataset(IRIS_DATASET_PATH)
    try:
        output = split_set(df)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to split dataset: {e}")
    
    

@router.post("/train-model", tags=["Model"])
async def train_model_endpoint(model_name: str = "LogisticRegression", target_column: str = "Species"):
    """
    Endpoint to train a classification model.
    
    Parameters:
        model_name (str): The name of the model to train.
        target_column (str): The target column to use for training.

    Returns:
        dict: A success message with the saved model's path.
    """
    try:
        data = process_dataset(IRIS_DATASET_PATH)
        data = process_dataset(IRIS_DATASET_PATH)

        output = split_set(data)
        
        model_path = train_model(output["train_data"], model_name=model_name, target_column=target_column)

        return {"message": "Model trained and saved successfully.", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {e}")

@router.post("/predict", tags=["Model"])
async def predict_endpoint(
    model_name: str = "LogisticRegression",
    input_data: List[Dict] = []):
    """
    Endpoint to make predictions using a trained model.
    [
        {
            "SepalLengthCm": 5.0,
            "SepalWidthCm": 3.4,
            "PetalLengthCm": 1.5,
            "PetalWidthCm": 0.2
        }
    ]
    Parameters:
        model_name (str): The name of the trained model to use for predictions.
        input_data (List[Dict]): Input features in JSON format.

    Returns:
        dict: Predictions made by the model.
    """
    try:
        model = load_trained_model(model_name)

        predictions = make_predictions(model, input_data)

        return {
            "message": "Predictions made successfully.",
            "predictions": predictions
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@router.get("/retrieve-parameters", tags=["Firestore"])
async def retrieve_parameters_endpoint():
    """
    Endpoint to retrieve parameters from Firestore.

    Returns:
        dict: The parameters retrieved from Firestore.
    """
    try:
        # Call the function to fetch parameters
        parameters = get_firestore_parameters()

        # Return success response with parameters
        return {
            "message": "Parameters retrieved successfully.",
            "parameters": parameters
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
