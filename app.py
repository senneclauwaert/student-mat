# Student Math Grade Prediction API
# This is a simple FastAPI application for predicting student math grades

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Student Math Grade Predictor",
    description="API to predict student math grades based on various factors",
    version="1.0.0"
)

# Load the trained model and preprocessing objects
try:
    model = joblib.load('best_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    print("Model and preprocessing objects loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please run the Jupyter notebook first to train and save the model.")

# Define input data model
class StudentData(BaseModel):
    school: str  # "GP" or "MS"
    sex: str  # "F" or "M"
    age: int  # 15 to 22
    address: str  # "U" or "R"
    famsize: str  # "LE3" or "GT3"
    Pstatus: str  # "T" or "A"
    Medu: int  # 0 to 4
    Fedu: int  # 0 to 4
    Mjob: str  # "teacher", "health", "services", "at_home", or "other"
    Fjob: str  # "teacher", "health", "services", "at_home", or "other"
    reason: str  # "home", "reputation", "course", or "other"
    guardian: str  # "mother", "father", or "other"
    traveltime: int  # 1 to 4
    studytime: int  # 1 to 4
    failures: int  # 0 to 4
    schoolsup: str  # "yes" or "no"
    famsup: str  # "yes" or "no"
    paid: str  # "yes" or "no"
    activities: str  # "yes" or "no"
    nursery: str  # "yes" or "no"
    higher: str  # "yes" or "no"
    internet: str  # "yes" or "no"
    romantic: str  # "yes" or "no"
    famrel: int  # 1 to 5
    freetime: int  # 1 to 5
    goout: int  # 1 to 5
    Dalc: int  # 1 to 5
    Walc: int  # 1 to 5
    health: int  # 1 to 5
    absences: int  # 0 to 93
    G1: int  # 0 to 20 (first period grade)
    G2: int  # 0 to 20 (second period grade)

# Define response model
class PredictionResponse(BaseModel):
    predicted_grade: float
    rounded_grade: int
    confidence_info: str

@app.get("/")
async def root():
    """Welcome message for the API"""
    return {
        "message": "Welcome to the Student Math Grade Predictor API",
        "description": "Use POST /predict to predict a student's final math grade",
        "documentation": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(student: StudentData):
    """
    Predict the final math grade (G3) for a student based on their characteristics
    """
    try:
        # Convert input to dictionary
        student_dict = student.model_dump()
        
        # Create DataFrame with the input
        input_df = pd.DataFrame([student_dict])
        
        # Encode categorical variables using the saved label encoders
        for col in label_encoders.keys():
            if col in input_df.columns:
                try:
                    # Handle unknown categories by using the most frequent class
                    if student_dict[col] not in label_encoders[col].classes_:
                        # Use the first class as default for unknown values
                        input_df[col] = 0
                    else:
                        input_df[col] = label_encoders[col].transform([student_dict[col]])[0]
                except Exception as e:
                    # Default to 0 if encoding fails
                    input_df[col] = 0
        
        # Reorder columns to match training features
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Check if model requires scaling (for KNN)
        model_type = type(model).__name__
        if model_type == "KNeighborsRegressor":
            # Scale the features for KNN
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            # Use unscaled features for tree-based models
            prediction = model.predict(input_df)[0]
        
        # Round prediction to nearest integer (grades are typically integers)
        rounded_prediction = round(prediction)
        
        # Ensure prediction is within valid range (0-20)
        rounded_prediction = max(0, min(20, rounded_prediction))
        
        return PredictionResponse(
            predicted_grade=float(prediction),
            rounded_grade=rounded_prediction,
            confidence_info=f"Prediction made using {model_type} model"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the trained model"""
    try:
        model_type = type(model).__name__
        return {
            "model_type": model_type,
            "features_count": len(features),
            "features": features,
            "categorical_encoders": list(label_encoders.keys()),
            "description": "Model trained on student-mat dataset to predict final math grades"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Example data endpoint
@app.get("/example")
async def get_example():
    """Get an example input for testing the API"""
    return {
        "example_input": {
            "school": "GP",
            "sex": "F",
            "age": 18,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "at_home",
            "Fjob": "teacher",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 2,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "yes",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "yes",
            "higher": "yes",
            "internet": "no",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 4,
            "Dalc": 1,
            "Walc": 1,
            "health": 3,
            "absences": 6,
            "G1": 5,
            "G2": 6
        },
        "description": "Copy this example and modify values to test the prediction endpoint"
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)