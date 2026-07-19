
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
target_names = joblib.load("target_names.pkl")

st.title("Iris Flower Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Input Data")
    st.write(data)

    X = scaler.transform(data)

    prediction = model.predict(X)

    data["Prediction"] = [target_names[i] for i in prediction]

    st.subheader("Prediction")
    st.write(data)

else:
    st.subheader("Manual Input")

    values = []

    for feature in feature_names:
        values.append(st.number_input(feature))

    if st.button("Predict"):
        X = np.array(values).reshape(1, -1)
        X = scaler.transform(X)
        pred = model.predict(X)

        st.success(f"Prediction : {target_names[pred[0]]}")
