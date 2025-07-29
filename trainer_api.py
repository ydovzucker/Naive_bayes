# trainer_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.dataLoader import DataLoader
from app.cleaner import Cleaner
from app.trainer import Trainer
from app.config import target_column

app = FastAPI()

@app.on_event("startup")
def train_model():
    try:
        loader = DataLoader()
        raw_data = loader.get_data()
        cleaner = Cleaner(raw_data)
        cleaned_data = cleaner.clean()

        trainer = Trainer(cleaned_data, target_column)
        prior, conditional_prob = trainer.train()

        # Save model data and cleaned data in app state for API access
        app.state.model_data = {
            "prior": prior,
            "conditional_prob": conditional_prob,
            "features": [col for col in cleaned_data.columns if col != target_column],
            "cleaned_data": cleaned_data.to_dict(orient="records")
        }

        print("✅ Model trained and ready.")
    except Exception as e:
        print(f"Startup failed: {e}")

@app.get("/")
def root():
    return {"message": "Trainer API is running."}

@app.get("/model")
def get_model_data():
    model_data = getattr(app.state, "model_data", None)
    if model_data is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return JSONResponse(content=model_data)