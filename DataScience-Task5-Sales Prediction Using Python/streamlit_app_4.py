
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
needs_scaling = joblib.load("needs_scaling.pkl")
feature_columns = joblib.load("feature_columns.pkl")
best_model_name = joblib.load("best_model_name.pkl")
metrics = joblib.load("metrics.pkl")
numeric_ranges = joblib.load("numeric_ranges.pkl")

st.title("Sales Prediction")

st.write("### Best Model")
st.success(best_model_name)

st.write("### Model Performance")
st.write(metrics)

prediction_type = st.radio(
    "Choose Prediction Type",
    ["Manual Input", "Upload CSV"]
)

# -----------------------
# Manual Prediction
# -----------------------

if prediction_type == "Manual Input":

    input_data = {}

    for col in feature_columns:
        min_val, max_val, default = numeric_ranges[col]

        input_data[col] = st.number_input(
            col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default)
        )

    if st.button("Predict"):

        df = pd.DataFrame([input_data])

        if needs_scaling:
            df = scaler.transform(df)

        prediction = model.predict(df)

        st.success(f"Predicted Sales : {prediction[0]:.2f}")

# -----------------------
# CSV Prediction
# -----------------------

else:

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(df.head())

        if list(df.columns) != feature_columns:

            st.error("CSV columns do not match training columns.")

        else:

            X = df.copy()

            if needs_scaling:
                X = scaler.transform(X)

            prediction = model.predict(X)

            df["Prediction"] = prediction

            st.write(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Predictions",
                csv,
                "Predictions.csv",
                "text/csv"
            )
