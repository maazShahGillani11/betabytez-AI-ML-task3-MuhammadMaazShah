import streamlit as st
import requests

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)
# Simple custom CSS for a cleaner interface (With correct streamlit argument)
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter student metrics below to estimate their final exam score using our trained ML model.</div>', unsafe_allow_html=True)
# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict"
# 2. Input Form Elements
st.subheader("Student Parameters")
with st.form("prediction_form"):
    weekly_self_study_hours = st.slider(
        "Weekly Self-Study Hours",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=0.5,
        help="How many hours does the student study weekly on their own?"
    )
    attendance_percentage = st.slider(
        "Attendance Percentage (%)",
        min_value=50.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        help="Overall class attendance percentage."
    )
    class_participation = st.slider(
        "Class Participation Rating (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        help="Active interaction rating in class from 0 to 10."
    )
    # Form Submit Button
    submit_btn = st.form_submit_button("Predict Score 🎯")
# 3. Handle Form Submission
if submit_btn:
    # Prepare JSON payload matching the FastAPI Pydantic Schema
    payload = {
        "weekly_self_study_hours": weekly_self_study_hours,
        "attendance_percentage": attendance_percentage,
        "class_participation": class_participation
    }
    try:
        # Send POST request to FastAPI backend
        with st.spinner("Calculating performance score..."):
            response = requests.post(API_URL, json=payload)   
        if response.status_code == 200:
            result = response.json()
            predicted_score = result["predicted_score"]
            confidence = result["confidence_score"]
            error_margin = result["error_margin_mae"]
            # Display Result visually
            st.success("🎉 Prediction Completed Successfully!")            
            # Create two columns to show Score and Confidence beautifully
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Estimated Final Exam Score", value=f"{predicted_score} / 100")
            with col2:
                st.metric(label="Model Confidence Score", value=confidence, delta=f"Error: {error_margin}", delta_color="inverse")
            # Add a visual progress/performance bar
            st.progress(predicted_score / 100)
        else:
            st.error(f"Backend Error (Status Code {response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection Failed! Please ensure your FastAPI backend is running at http://127.0.0.1:8000")