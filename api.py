# Simple Student Grade Prediction API (First-Year Style)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# 1) Create the app
app = FastAPI(
    title="Student Grade Predictor (First-Year Style)",
    description="Predicts final math grade (G3) using a trained model from the notebook.",
    version="1.0.0"
)

# 2) Load model + preprocessors
try:
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    print("Loaded model, scaler, and label encoders.")
except Exception as e:
    print("Could not load one of the required files:", e)
    model = None
    scaler = None
    label_encoders = {}

# Try to load exact feature order; else fallback
try:
    feature_names = joblib.load("feature_names.joblib")
    print("Loaded feature names used during training.")
except Exception:
    print("feature_names.joblib not found. Using a fallback list of columns.")
    feature_names = [
        'school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob',
        'reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid',
        'activities','nursery','higher','internet','romantic','famrel','freetime','goout',
        'Dalc','Walc','health','absences','G1','G2'
    ]

# 3) Input schema (simple defaults)
class StudentData(BaseModel):
    G2: int
    G1: int
    failures: int = 0
    age: int = 16
    absences: int = 0
    studytime: int = 2
    sex: str = "F"
    school: str = "GP"
    address: str = "U"
    famsize: str = "GT3"
    Pstatus: str = "T"
    Mjob: str = "other"
    Fjob: str = "other"
    reason: str = "course"
    guardian: str = "mother"
    Medu: int = 3
    Fedu: int = 3
    traveltime: int = 1
    schoolsup: str = "no"
    famsup: str = "yes"
    paid: str = "no"
    activities: str = "no"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "yes"
    romantic: str = "no"
    famrel: int = 4
    freetime: int = 3
    goout: int = 3
    Dalc: int = 1
    Walc: int = 1
    health: int = 3

# Rename field to avoid pydantic "model_" warning
class PredictionOut(BaseModel):
    predicted_grade: float
    rounded_grade: int
    model_used: str

@app.get("/")
def root():
    return {
        "message": "Welcome to the Student Grade Predictor (First-Year Style)!",
        "how_to_use": "POST to /predict with a JSON body. Try /docs for an easy UI.",
        "example": {
            "G2": 16, "G1": 15, "failures": 0, "age": 16, "absences": 2, "studytime": 3,
            "sex": "F", "school": "GP", "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
            "Medu": 3, "Fedu": 3, "traveltime": 1, "schoolsup": "no", "famsup": "yes",
            "paid": "no", "activities": "no", "nursery": "yes", "higher": "yes",
            "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3,
            "Dalc": 1, "Walc": 1, "health": 3
        }
    }

def simple_encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if isinstance(label_encoders, dict):
        for col, le in label_encoders.items():
            if col in df2.columns:
                try:
                    val = df2[col].iloc[0]
                    if val not in getattr(le, "classes_", []):
                        df2[col] = 0
                    else:
                        df2[col] = le.transform(df2[col])
                except Exception:
                    df2[col] = 0
    return df2

@app.post("/predict", response_model=PredictionOut)
def predict(student: StudentData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Train and save them first.")

    row = pd.DataFrame([student.dict()])
    row = row.reindex(columns=feature_names, fill_value=0)
    row_encoded = simple_encode_dataframe(row)
    X_scaled = scaler.transform(row_encoded.values)
    y_pred = model.predict(X_scaled)[0]

    rounded = int(round(float(y_pred)))
    rounded = max(0, min(20, rounded))

    return PredictionOut(
        predicted_grade=float(y_pred),
        rounded_grade=rounded,
        model_used=type(model).__name__
    )

if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: file is named api.py, so use "api:app"
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
