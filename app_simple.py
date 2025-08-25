# Simple Student Grade Prediction Web Service
# This is the EASY version - only needs 6 pieces of student information
# Use this one for testing!

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Student Grade Predictor (Easy Version)",
    description="Predicts student math grades using only 6 important things about the student",
    version="1.0.0"
)

# Load the trained computer model (the brain that predicts grades)
try:
    model = joblib.load('best_model.pkl')                    # The trained AI model
    label_encoders = joblib.load('label_encoders.pkl')       # Converts text to numbers
    scaler = joblib.load('scaler.pkl')                       # Makes numbers similar size
    features = joblib.load('features.pkl')                   # List of student info we use
    print("✅ AI model loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: Cannot find model files!")
    print("👉 Please run the Jupyter notebook first to train the AI model.")

# What student information we need (only the most important 6 things!)
class StudentDataSimple(BaseModel):
    # REQUIRED - These 6 things are most important for predicting grades:
    G2: int         # Last test grade (0-20) - VERY important!
    G1: int         # First test grade (0-20) - Also very important!
    failures: int = 0    # How many classes failed before (0-4)
    age: int = 16        # Student age (15-22)
    absences: int = 0    # Days missed school (0-93)
    studytime: int = 2   # Hours studied per week (1=little, 2=some, 3=good, 4=a lot)
    
    # Optional - These help but are not required:
    Medu: int = 3   # Mom's education level (0-4, higher=more education)
    Fedu: int = 3   # Dad's education level (0-4, higher=more education)
    goout: int = 3  # How much student goes out with friends (1-5)
    Dalc: int = 1   # Alcohol on school days (1=none, 5=a lot)
    
    # Optional fields with sensible defaults
    sex: str = "F"  # "F" or "M"
    school: str = "GP"  # "GP" or "MS"
    address: str = "U"  # "U" (urban) or "R" (rural)
    famsize: str = "GT3"  # "GT3" or "LE3"
    Pstatus: str = "T"  # Parents together: "T" or "A" (apart)
    Mjob: str = "other"  # Mother's job
    Fjob: str = "other"  # Father's job
    reason: str = "course"  # Reason for school choice
    guardian: str = "mother"  # Guardian
    traveltime: int = 1  # Travel time to school (1-4)
    schoolsup: str = "no"  # Extra educational support
    famsup: str = "yes"  # Family educational support
    paid: str = "no"  # Extra paid classes
    activities: str = "no"  # Extra-curricular activities
    nursery: str = "yes"  # Attended nursery
    higher: str = "yes"  # Wants higher education
    internet: str = "yes"  # Internet at home
    romantic: str = "no"  # In romantic relationship
    famrel: int = 4  # Family relationships quality (1-5)
    freetime: int = 3  # Free time after school (1-5)
    Walc: int = 1  # Weekend alcohol consumption (1-5)
    health: int = 3  # Health status (1-5)

# Define response model
class PredictionResponse(BaseModel):
    predicted_grade: float
    rounded_grade: int
    confidence_info: str
    input_summary: dict

@app.get("/")
async def root():
    """Welcome message for the easy API"""
    return {
        "message": "🎓 Welcome to the Student Grade Predictor (Easy Version)!",
        "description": "Only needs 6 important things about the student",
        "what_you_need": ["G2 (last test grade)", "G1 (first test grade)", "failures", "age", "absences", "studytime"],
        "how_to_test": "Go to /docs to test the prediction",
        "example_data": "Go to /example to see sample student data"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_grade(student: StudentDataSimple):
    """
    Predict student's final math grade
    Just send student information and get back the predicted grade!
    """
    try:
        # Step 1: Convert student info to format the AI can understand
        student_dict = student.model_dump()
        
        # Step 2: Put student data in a table format
        input_df = pd.DataFrame([student_dict])
        
        # Step 3: Convert text to numbers (AI only understands numbers)
        for col in label_encoders.keys():
            if col in input_df.columns:
                try:
                    # Convert things like "male"/"female" to numbers like 0/1
                    if student_dict[col] not in label_encoders[col].classes_:
                        input_df[col] = 0  # Use 0 if we don't recognize the value
                    else:
                        input_df[col] = label_encoders[col].transform([student_dict[col]])[0]
                except Exception as e:
                    input_df[col] = 0  # Use 0 if something goes wrong
        
        # Step 4: Make sure data is in the right order and format
        input_df = input_df.reindex(columns=features, fill_value=0)
        
        # Step 5: Ask the AI to predict the grade!
        model_type = type(model).__name__
        if model_type == "KNeighborsRegressor":
            # Some AI models need numbers to be the same size
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            # Most AI models can use the data directly
            prediction = model.predict(input_df)[0]
        
        # Step 6: Clean up the prediction (make sure it's a valid grade)
        rounded_prediction = round(prediction)  # Round to whole number
        rounded_prediction = max(0, min(20, rounded_prediction))  # Keep between 0-20
        
        return PredictionResponse(
            predicted_grade=float(prediction),
            rounded_grade=rounded_prediction,
            confidence_info=f"Prediction made using {model_type} model",
            input_summary={
                "previous_grades": f"G1: {student.G1}, G2: {student.G2}",
                "key_factors": f"Failures: {student.failures}, Age: {student.age}, Absences: {student.absences}",
                "study_habits": f"Study time: {student.studytime}/4"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

@app.get("/example")
async def get_example():
    """Get example student data for testing"""
    return {
        "good_student": {
            "G2": 16,    # Good last test grade
            "G1": 15,    # Good first test grade  
            "failures": 0,   # Never failed a class
            "age": 16,       # Normal age
            "absences": 1,   # Rarely misses school
            "studytime": 4   # Studies a lot
        },
        "struggling_student": {
            "G2": 8,     # Poor last test grade
            "G1": 7,     # Poor first test grade
            "failures": 2,   # Failed 2 classes before
            "age": 18,       # Older (maybe repeated grades)
            "absences": 12,  # Misses school often
            "studytime": 1   # Doesn't study much
        },
        "how_to_use": "Copy one of these examples and paste it in the /predict endpoint at /docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)