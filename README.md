# 🎓 Student Performance Prediction System (End-to-End Regression)

An enterprise-grade, end-to-end Machine Learning prediction system built using Scikit-Learn, FastAPI, and Streamlit. The system predicts a student's final score out of 100 based on continuous study metrics, complete with strict data validation and a production confidence score.

# Project Description
This repository contains the deliverables for Week 3–4 of the BetaBytez AI/ML Program 2026. The objective of this task is to transition from a localized notebook environment into a multi-tier production service. The architecture features a trained regression model wrapped inside a FastAPI backend, exposing an endpoint to serve predictions with a calculated validation confidence metric to a Streamlit frontend client.

# Student Performance & Exam Score Predictor
An end-to-end Machine Learning web application built using Python, Scikit-Learn, FastAPI, and Streamlit to evaluate and estimate a student's final examination score based on continuous academic engagement metrics.

1. Dataset Overview
  The project utilizes a large-scale student performance tracking dataset.
  Target Variable: `total_score` (Continuous Numerical Bounds: 0 to 100) - Indicates the final score achieved by the student in the examination block.
  Features: Key metrics tracking student behavior:
  `weekly_self_study_hours` (Continuous hours spent studying independently).
  `attendance_percentage` (Overall class attendance history).
  `class_participation` (Active interaction rating scored from 0 to 10).

# 2. Engineering Approach
  A. Exploratory Data Analysis (EDA) & Cleaning
  Analyzed summary statistics across 1,000,000 student data records.
  Dropped unique identifiers (`student_id`) and post-score derived evaluation labels (`grade`) to mitigate data leakage and keep training features clean.
  Handled outliers and verified zero missing data points (`isnull().sum() == 0`) across core columns before splitting features.
  B. Preprocessing & Feature Engineering
  Feature Separation: Structured input arrays strictly containing numerical metrics (`weekly_self_study_hours`, `attendance_percentage`, `class_participation`).
  Train-Test Split: Apportioned the unified cleansed dataset into an 80% Training block and a 20% Testing partition (`random_state=42`) to maintain benchmarking stability.

  C. Model Training & Evaluation
  Trained and cross-compared three distinct regression approaches under supervised metrics (MAE and R^2 Score):
  1. Random Forest Regressor: Selected for production deployment. It demonstrated robust variance control and achieved the lowest Mean Absolute Error (\text{MAE} \approx \pm1.5\text{ marks}).
  2. Linear Regression: Evaluated as a baseline mathematical model to map linear correlation.
  3. Decision Tree Regressor: Tested with deep branch variations but controlled to avoid overfitting.

# 3. How to Run the Web Application Locally
# Prerequisites
Make sure you have Python 3.8+ installed on your system.
# Step-by-Step Setup
1. Clone the Repository:
   git clone https://github.com/maazShahGillani11/betabytez-AI-ML-task3-MuhammadMaazShah.git
   cd betabytez-AI-ML-task3-MuhammadMaazShah

2. Install the Required Packages:
   pip install pandas numpy scikit-learn fastapi uvicorn streamlit requests seaborn matplotlib

3. Launch the Backend REST API Server:
   cd api
   uvicorn app:app --reload
   Server loads the model (`best_student_model.pkl`) and runs locally at http://127.0.0.1:8000
   Access interactive endpoint testing sheets via http://127.0.0.1:8000/docs

4. Launch the Frontend Web UI Client:
   Open a separate terminal instance/tab and execute:
   cd frontend
   streamlit run appUI.py
   Navigate to the interactive UI panel at http://localhost:8501   

# Repository Structure
```text
├── model/
│   ├── Student_Performance.csv      # Dataset (1 Million Records)
│   ├── student_regression.ipynb     # Jupyter Notebook (Training, Plots & Comparison)
│   └── best_student_model.pkl       # Trained Random Forest Regressor Model
├── api/
│   ├── app.py                       # Upgraded FastAPI Backend Application
│   └── best_student_model.pkl       # Copied Model Object for Production API
├── frontend/
│   └── appUI.py                     # Streamlit Interactive User Interface
└── README.md                        # Project Documentation & Setup Guide