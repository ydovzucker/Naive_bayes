from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
import pandas as pd

from app.dataLoader import DataLoader
from app.cleaner import Cleaner
from app.trainer import Trainer
from app.modelChecker import ModelChecker
from app.config import target_column

app = FastAPI()

class SampleInput(BaseModel):
    sample: Dict[str, str]

@app.on_event("startup")
def startup_event():
    try:
        # Load and clean data
        loader = DataLoader()
        raw_data = loader.get_data()
        cleaner = Cleaner(raw_data)
        cleaned_data = cleaner.clean()

        # Train model
        trainer = Trainer(cleaned_data, target_column)
        prior, conditional_prob = trainer.train()
        model = ModelChecker(prior, conditional_prob, target_column)

        # Store everything in app.state
        app.state.cleaned_data = cleaned_data
        app.state.model = model
        app.state.feature_names = [col for col in cleaned_data.columns if col != target_column]

        print("✅ Model trained successfully on startup.")

    except Exception as e:
        print(f"❌ Startup failed: {e}")

@app.get("/")
def root():
    return {"message": "Naive Bayes API is running."}

@app.get("/features")
def get_features(request: Request):
    features = getattr(request.app.state, "feature_names", None)
    if not features:
        raise HTTPException(status_code=500, detail="Model not trained.")
    return {"features": features}

@app.post("/predict")
def predict(input: SampleInput, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained.")

    try:
        prediction = model.predict(input.sample)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/accuracy")
def get_accuracy(request: Request):
    model = getattr(request.app.state, "model", None)
    data = getattr(request.app.state, "cleaned_data", None)

    if model is None or data is None:
        raise HTTPException(status_code=500, detail="Model or data not available.")

    correct = 0
    for _, row in data.iterrows():
        sample = row.drop(target_column).to_dict()
        actual = row[target_column]
        pred = model.predict(sample)
        if pred == actual:
            correct += 1

    accuracy = correct / len(data)
    return {"accuracy": round(accuracy, 4)}

@app.get("/confusion_matrix")
def get_confusion_matrix(request: Request):
    model = getattr(request.app.state, "model", None)
    data = getattr(request.app.state, "cleaned_data", None)

    if model is None or data is None:
        raise HTTPException(status_code=500, detail="Model or data not available.")

    labels = sorted(data[target_column].unique())
    matrix = {label: {l: 0 for l in labels} for label in labels}

    for _, row in data.iterrows():
        sample = row.drop(target_column).to_dict()
        true_label = row[target_column]
        pred = model.predict(sample)
        matrix[true_label][pred] += 1

    return {"confusion_matrix": matrix}