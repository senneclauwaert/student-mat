# Student Grade Prediction - First-Year Student Project

This project predicts how well students will do in math class using machine learning. Written in a simple, first-year student style.

## What Files Do What

- `Assignment.ipynb` - Main notebook that trains 3 different AI models to predict grades
- `api.py` - Simple web service API to use the trained model 
- `requirements.txt` - List of programs we need to install
- `student-mat.csv` - Data about 395 students from Portugal
- `metadata.txt` - Explains what each piece of student information means
- Model files (created after training): `best_model.joblib`, `scaler.joblib`, `label_encoders.joblib`

## How to Run Everything

### Step 1: Set Up Your Computer
```bash
# Make a clean workspace
python -m venv venv

# Turn on the workspace
# Windows users:
venv\Scripts\activate
# Mac/Linux users:
source venv/bin/activate

# Install needed programs
pip install -r requirements.txt
```

### Step 2: Train the AI Models
1. Open `Assignment.ipynb` in Jupyter Notebook or VS Code
2. Run ALL the code cells from top to bottom (takes a few minutes)
3. This trains 3 different models: Decision Tree, KNN, and Random Forest
4. It automatically picks the best model and saves it as `.joblib` files



### Step 3: Start the Web API
```bash
python api.py
```

Open your web browser and go to `http://localhost:8000`


## How to Test It

### Quick Test Example:
Send this student information to get a grade prediction:
```json
{
  "G2": 15,
  "G1": 13, 
  "failures": 0,
  "age": 17,
  "absences": 2,
  "studytime": 3
}
```
(The API will fill in default values for the other student information)

### What These Numbers Mean:
- **G2**: Grade from last test (0-20)
- **G1**: Grade from first test (0-20)
- **failures**: How many classes failed before (0-4)
- **age**: Student age (15-22)
- **absences**: Days missed school (0-93)
- **studytime**: Hours studied per week (1=little, 4=a lot)

### How to Send a Test:
1. Go to `http://localhost:8000/docs` in your browser
2. Click on "POST /predict"
3. Click "Try it out"
4. Copy the example above into the box
5. Click "Execute"
6. See the predicted grade!

## What You Get Back

The API returns:
- **predicted_grade**: Exact prediction (like 14.237)
- **rounded_grade**: Rounded number (like 14)
- **model_used**: Which AI model made the prediction (DecisionTreeRegressor, etc.)

## About the AI Models

**What the models do:**
- **Input**: Student info like past grades, study time, family background
- **Output**: Predicted final math grade (0-20 scale)
- **Training data**: 395 real students from Portugal
- **Accuracy**: Usually within 1-2 grade points

**Three models tested:**
1. **Decision Tree** - Makes simple yes/no decisions 
2. **K-Nearest Neighbors (KNN)** - Looks at similar students
3. **Random Forest** - Combines many decision trees (often the best)

**Complete Steps:**
1. ✅ Run the notebook to train models
2. ✅ Install requirements in virtual environment  
3. ✅ Start the API with `python api.py`
4. ✅ Test at `http://localhost:8000/docs`
