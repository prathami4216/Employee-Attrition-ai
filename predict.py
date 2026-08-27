"""
=========================================================
Employee Attrition Prediction System
Prediction Module
=========================================================
Supports
✓ Single Employee Prediction
✓ Batch Prediction
✓ Confidence Score
✓ CSV / Excel Prediction
=========================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

from config import *

from utils.preprocess import (
    load_pipeline,
    load_encoder,
    preprocess_employee,
    load_dataset
)

# ==========================================================
# Load Model
# ==========================================================

def load_model():

    """
    Load trained Decision Tree pipeline.
    """

    print("\nLoading trained model...")

    pipeline = load_pipeline()

    print("Model Loaded Successfully.")

    return pipeline


# ==========================================================
# Load Label Encoder
# ==========================================================

def load_label_encoder():

    """
    Load target encoder.
    """

    encoder = load_encoder()

    return encoder


# ==========================================================
# Predict Single Employee
# ==========================================================

def predict_employee(employee_data):

    """
    employee_data -> dictionary
    """

    model = load_model()

    encoder = load_label_encoder()

    employee = preprocess_employee(employee_data)

    prediction = model.predict(employee)[0]

    label = encoder.inverse_transform([prediction])[0]

    probability = model.predict_proba(employee)[0]

    confidence = round(max(probability) * 100, 2)

    risk = confidence

    return {
    "prediction": label,
    "confidence": confidence,
    "risk_percentage": confidence,
    "probability": probability.tolist(),
    "employee_data": employee_data,
    "risk_level": (
        "High"
        if confidence >= 80 else
        "Medium"
        if confidence >= 60 else
        "Low"
    )
}


# ==========================================================
# Pretty Print Result
# ==========================================================

def print_prediction(result):

    print("\n")

    print("=" * 60)

    print("EMPLOYEE ATTRITION PREDICTION")

    print("=" * 60)

    print("Prediction       :", result["prediction"])

    print("Confidence       :", f"{result['confidence']} %")

    print("Risk Percentage  :", f"{result['risk_percentage']} %")

    print("=" * 60)
    
# ==========================================================
# Batch Prediction
# ==========================================================

def predict_dataset(file_path=None):

    """
    Predict attrition for an entire CSV/XLSX dataset.
    """

    model = load_model()

    encoder = load_label_encoder()

    df = load_dataset(file_path)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    employee_ids = None

    if "Employee_ID" in df.columns:
        employee_ids = df["Employee_ID"]
        df = df.drop(columns=["Employee_ID"])

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)

    labels = encoder.inverse_transform(predictions)

    confidence = np.max(probabilities, axis=1) * 100

    results = df.copy()

    if employee_ids is not None:
        results.insert(0, "Employee_ID", employee_ids)

    results["Prediction"] = labels
    results["Confidence (%)"] = confidence.round(2)

    return results


# ==========================================================
# Save Predictions
# ==========================================================

def save_predictions(results, filename="prediction_results.csv"):

    extension = os.path.splitext(filename)[1].lower()

    if extension == ".xlsx":

        results.to_excel(
            filename,
            index=False
        )

    else:

        results.to_csv(
            filename,
            index=False
        )

    print("\nPredictions saved successfully.")
    print(filename)


# ==========================================================
# Prediction Summary
# ==========================================================

def prediction_summary(results):

    print("\n")
    print("="*60)
    print("PREDICTION SUMMARY")
    print("="*60)

    total = len(results)

    yes = (results["Prediction"] == "Yes").sum()
    no = (results["Prediction"] == "No").sum()


    print("Total Employees :", total)
    print("Likely to Leave :", yes)
    print("Likely to Stay  :", no)

    print("="*60)


# ==========================================================
# Test Module
# ==========================================================

if __name__ == "__main__":

    sample_employee = {

        "Age":35,
        "Gender":"Male",
        "Marital_Status":"Single",
        "Department":"IT",
        "Job_Role":"Analyst",
        "Job_Level":2,
        "Monthly_Income":7500,
        "Hourly_Rate":55,
        "Years_at_Company":5,
        "Years_in_Current_Role":3,
        "Years_Since_Last_Promotion":1,
        "Work_Life_Balance":3,
        "Job_Satisfaction":4,
        "Performance_Rating":3,
        "Training_Hours_Last_Year":25,
        "Overtime":"Yes",
        "Project_Count":5,
        "Average_Hours_Worked_Per_Week":46,
        "Absenteeism":2,
        "Work_Environment_Satisfaction":4,
        "Relationship_with_Manager":3,
        "Job_Involvement":4,
        "Distance_From_Home":10,
        "Number_of_Companies_Worked":2

    }

    result = predict_employee(sample_employee)

    print_prediction(result)

    print("\nDo you want batch prediction? (y/n)")

    choice = input("> ").strip().lower()

    if choice == "y":

        results = predict_dataset()

        prediction_summary(results)

        save_predictions(
            results,
            "prediction_results.csv"
        )

        print(results.head())    