from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from trainer import Trainer
from cleaner import Cleaner
from modelChecker import ModelChecker

app = FastAPI()

# Global variables to store state after training
trained_checker = None
feature_names = []
target_column = ""

# === Train Endpoint ===
class TrainRequest(BaseModel):
    data: list[dict]
    target_column: str

@app.post("/train")
def train_model(request: TrainRequest):
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

# === Predict Endpoint ===
@app.post("/predict")
def predict(sample: Dict[str, str]):
    if trained_checker is None:
        return {"error": "Model not trained yet. Please train first using /train."}
    prediction = trained_checker.predict(sample)
    return {"prediction": prediction}