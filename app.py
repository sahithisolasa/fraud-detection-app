import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is **fraudulent or legitimate** using a trained ML model.")

# Sidebar input
st.sidebar.header("Transaction Input Features")

# Collect input data from user
def user_input_features():
    time = st.sidebar.number_input("Time (in seconds)", min_value=0)
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
    
    # 28 anonymized features like V1-V28 (for demo, allow small manual input)
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0)
    
    data = pd.DataFrame([features])
    
    # Add scaled amount and time
    data['scaled_amount'] = scaler.transform(np.array(amount).reshape(-1,1))
    data['scaled_time'] = scaler.transform(np.array(time).reshape(-1,1))
    
    return data

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "🚨 Fraud Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction"
    st.subheader("Prediction Result:")
    st.write(result)
