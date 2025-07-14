import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model dictionary and extract model
with open("customer_churn_model.pkl", "rb") as f:
    model_dict = pickle.load(f)
model = model_dict["model"]

# Try to get feature names from model
try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error("❌ Feature names not found in model. Make sure model was trained with a DataFrame.")
    st.stop()

# Load encoder dictionary
with open("encoder.pkl", "rb") as f:
    encoder_dict = pickle.load(f)

st.title("📊 Customer Churn Prediction App")

# User input
input_data = {
    'gender': st.selectbox("Gender", ["Male", "Female"]),
    'SeniorCitizen': st.selectbox("Senior Citizen", [0, 1]),
    'Partner': st.selectbox("Partner", ["Yes", "No"]),
    'Dependents': st.selectbox("Dependents", ["Yes", "No"]),
    'tenure': st.slider("Tenure (months)", 0, 72, 12),
    'PhoneService': st.selectbox("Phone Service", ["Yes", "No"]),
    'MultipleLines': st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    'InternetService': st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    'OnlineSecurity': st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    'OnlineBackup': st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    'DeviceProtection': st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    'TechSupport': st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    'StreamingTV': st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    'StreamingMovies': st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    'Contract': st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    'PaperlessBilling': st.selectbox("Paperless Billing", ["Yes", "No"]),
    'PaymentMethod': st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    'MonthlyCharges': st.number_input("Monthly Charges", 0.0, 200.0, 70.0),
    'TotalCharges': st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode using your label encoders
for col, encoder in encoder_dict.items():
    if col in input_df.columns:
        input_df[col] = encoder.transform(input_df[col])

# Reorder input to match model’s training columns
try:
    input_df = input_df[feature_names]
except KeyError as e:
    st.error(f"Feature mismatch error: {e}")
    st.stop()

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
