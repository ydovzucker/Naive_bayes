from fastapi import FastAPI
from pydantic import BaseModel
from dataLoader import DataLoader
from cleaner import Cleaner
from trainer import Trainer
from modelChecker import ModelChecker

app = FastAPI()
@app.post("/predict")
def predict(sample: Dict[str, str]):
    # We'll complete this logic in the next step
    return {"received": sample}