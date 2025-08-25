# Student Grade Prediction - Simple Guide

This project helps predict how well students will do in math class. It uses a computer program (AI) to guess student grades.

## What Files Do What

- `Assignment.ipynb` - Main notebook that trains the computer to predict grades
- `app.py` - Web service that needs ALL student information (32 things)
- `app_simple.py` - Web service that only needs 6 important things (EASIER!)
- `requirements.txt` - List of programs we need to install
- `student-mat.csv` - Data about 395 students
- `metadata.txt` - Explains what each piece of student information means

## How to Run Everything

### Step 1: Train the Computer Program
1. Open `Assignment.ipynb` in VS Code
2. Run ALL the code cells from top to bottom
3. This teaches the computer how to predict grades
4. It will save the best trained model

### Step 2: Set Up Your Computer
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

### Step 3: Start the Web Service

**Easy Version (recommended):**
```bash
python app_simple.py
```

**Full Version (harder):**
```bash
python app.py
```

Open your web browser and go to `http://localhost:8000`

**💡 Tip: Use the easy version first!**

## How to Test It

### Test with Easy Version:
Send this information to predict a grade:
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

The computer will tell you:
- **predicted_grade**: Exact prediction (like 14.2)
- **rounded_grade**: Rounded number (like 14)
- **confidence_info**: Which model made the prediction

## About the Computer Program

- **Looks at**: Student info like past grades, study time, family background
- **Predicts**: Final math grade (0-20 scale)
- **Trained on**: 395 real students from Portugal
- **Accuracy**: Usually within 1-2 grade points

## All Student Information Used

The computer looks at these things about each student:
- Personal: age, gender, where they live
- Family: parents' jobs and education
- School: study time, past grades, absences
- Social: free time, relationships

See `metadata.txt` for complete list.

## Need Help?

**Common Problems:**
- **"Model files not found"**: Run the notebook first!
- **"Import error"**: Make sure venv is active and requirements installed
- **"Connection error"**: Check if the web service is running

**Testing Steps:**
1. Train model in notebook ✅
2. Install requirements ✅
3. Start web service ✅
4. Test in browser ✅

That's it! You now have a working grade prediction system.