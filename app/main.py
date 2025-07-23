"""
FastAPI application for training and predicting using a Naive Bayes classifier.

This module defines two main endpoints:
- /train: Accepts labeled data, cleans it, and trains a Naive Bayes model.
- /predict: Accepts a single sample and returns the predicted class using the trained model.

Modules used:
- FastAPI: For building the API.
- Pydantic: For request data validation.
- pandas: For data manipulation.
- Trainer, Cleaner, ModelChecker: Internal modules handling ML logic.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from app.trainer import Trainer
from app.cleaner import Cleaner
from app.modelChecker import ModelChecker

app = FastAPI()

# === Global state ===
trained_checker = None  # Will hold the trained ModelChecker instance
feature_names = []  # Stores names of features (not including target)
target_column = ""  # Target column used during training


class TrainRequest(BaseModel):
    """
    Pydantic model to validate incoming JSON for training.

    Attributes:
        data (list[dict]): List of feature dictionaries for each sample.
        target_column (str): The name of the target/label column.
    """
    data: list[dict]
    target_column: str


@app.post("/train")
def train_model(request: TrainRequest):
    """
    Train a Naive Bayes model using the provided data.

    Args:
        request (TrainRequest): JSON payload with data and target_column.

    Returns:
        dict: A success message and the names of the features used.
    """
    global trained_checker, feature_names, target_column

    df = pd.DataFrame(request.data)
    target_column = request.target_column

    cleaner = Cleaner(df)
    cleaned_data = cleaner.clean()

    trainer = Trainer(cleaned_data, target_column)
    prior, conditional_prob = trainer.train()

    trained_checker = ModelChecker(prior, conditional_prob, target_column)

    feature_names = [col for col in df.columns if col != target_column]

    return {"message": "Model trained successfully", "features": feature_names}


@app.post("/predict")
def predict(sample: Dict[str, str]):
    """
    Predict the class of a new sample using the trained model.

    Args:
        sample (Dict[str, str]): A dictionary of feature values.

    Returns:
        dict: A dictionary with the predicted class label or an error message.
    """
    if trained_checker is None:
        return {"error": "Model not trained yet. Please train first using /train."}

    prediction = trained_checker.predict(sample)
    return {"prediction": prediction}