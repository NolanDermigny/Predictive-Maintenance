import streamlit as st
import pandas as pd
import joblib

st.title("Predictive Maintenance Dashboard")

model = joblib.load("models/rf_model.joblib")

uploaded_file = st.file_uploader("Upload sensor data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    if 'label' in df.columns:
        X = df.drop(columns=['label', 'timestamp'])
    else:
        X = df.drop(columns=['timestamp'])

    preds = model.predict(X)
    st.write("Predictions", preds)
