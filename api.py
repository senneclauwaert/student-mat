# Simple Student Grade Prediction API (Full Pipeline)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib
import numpy as np

app = FastAPI(title="Final Grade Predictor (Full Pipeline)")

try:
    model = joblib.load("best_model.joblib")  # full sklearn Pipeline
    feature_names = joblib.load("feature_names.joblib")
    print("✅ Model and feature names loaded.")
except Exception as e:
    model = None
    feature_names = None
    print("❌ Could not load model:", e)

class StudentInput(BaseModel):
    # Minimal example inputs; you can add more fields present in your X
    G1: int
    G2: int
    absences: int = 0
    studytime: int = 2
    failures: int = 0
    age: int = 16
    sex: str = "F"
    school: str = "GP"
    address: str = "U"

@app.get("/")
def root():
    return {
        "message": "Welcome. POST to /predict. Try /docs for an interactive UI.",
        "note": "Model is a full sklearn Pipeline (preprocessing + estimator)."
    }

@app.post("/predict")
def predict(item: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train & save it first.")
    df = pd.DataFrame([item.dict()])
    # Ensure expected columns exist; missing -> NaN (imputed by pipeline)
    if feature_names is not None:
        df = df.reindex(columns=feature_names, fill_value=np.nan)
    y_hat = float(model.predict(df)[0])
    return {"predicted_grade": y_hat, "rounded_grade": max(0, min(20, int(round(y_hat))))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)
