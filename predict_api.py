from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
from app.modelChecker import ModelChecker

app = FastAPI()

TRAINING_SERVER_URL = "http://trainer-api:8000/model" # name of the container in docker-compose

class InputData(BaseModel):
    data: dict

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Get the trained model from the trainer server
        response = requests.get(TRAINING_SERVER_URL)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve model from training server")

        model_data = response.json()
        prior = model_data["prior"]
        conditional_prob = model_data["conditional_prob"]
        features = model_data["features"]

        # Prepare the input for prediction
        df = pd.DataFrame([input_data.data])
        df = df[features]  # only the features used in training

        # Make the prediction
        checker = ModelChecker(prior, conditional_prob)
        prediction = checker.predict(df.iloc[0])

        return {"prediction": prediction}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
@app.get("/")
def root():
        return {"message": "Naive Bayes Predictor API running. Version: V2"}