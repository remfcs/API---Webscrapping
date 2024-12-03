import json
import os
from fastapi import APIRouter, HTTPException
import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from src.services.data import *

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



