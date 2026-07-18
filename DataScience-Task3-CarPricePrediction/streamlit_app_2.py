
import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load Saved Files
# -------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
model_name = joblib.load("best_model_name.pkl")
metrics = joblib.load("metrics.pkl")

st.title("🚗 Car Price Prediction")

st.write(f"### Best Model : {model_name}")

st.write("### Model Performance")
st.write(metrics)

# -------------------------
# Choose Input Method
# -------------------------

input_method = st.radio(
    "Choose Input Method",
    ["Upload CSV", "Manual Input"]
)

# ==========================================================
# MANUAL INPUT
# ==========================================================

if input_method == "Manual Input":

    st.subheader("Enter Car Details")

    name = st.text_input(
        "Car Name",
        "Maruti Swift Dzire VDI"
    )

    year = st.number_input(
        "Year",
        min_value=1990,
        max_value=2030,
        value=2015
    )

    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Diesel", "Petrol", "CNG", "LPG"]
    )

    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual", "Trustmark Dealer"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

    mileage = st.number_input(
        "Mileage",
        value=20.0
    )

    engine = st.number_input(
        "Engine (CC)",
        value=1248
    )

    max_power = st.number_input(
        "Max Power",
        value=74.0
    )

    seats = st.number_input(
        "Seats",
        value=5
    )

    if st.button("Predict Price"):

        df = pd.DataFrame({
            "name": [name],
            "year": [year],
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })

        # Convert transmission to numeric
        df["transmission"] = df["transmission"].map({
            "Manual": 0,
            "Automatic": 1
        })

        # Convert owner to numeric
        owner_map = {
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5
        }

        df["owner"] = df["owner"].map(owner_map)

        # Feature Engineering
        df["Brand"] = df["name"].str.split().str[0]

        df = pd.get_dummies(
            df,
            columns=["fuel", "seller_type"],
            drop_first=True,
            dtype=int
        )

        df = pd.get_dummies(
            df,
            columns=["Brand"],
            drop_first=True,
            dtype=int
        )

        # Remove unused columns
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

        if "seats" in df.columns:
            df.drop(columns=["seats"], inplace=True)

        # Match Training Features
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # Scaling
        X = scaler.transform(df)

        # Prediction
        prediction = model.predict(X)

        st.success(f"Predicted Car Price: ₹ {prediction[0]:,.2f}")

# ==========================================================
# CSV UPLOAD
# ==========================================================

elif input_method == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Convert transmission to numeric
        df["transmission"] = df["transmission"].map({
            "Manual": 0,
            "Automatic": 1
        })

        # Convert owner to numeric
        owner_map = {
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5
        }

        df["owner"] = df["owner"].map(owner_map)

        st.subheader("Uploaded Data")
        st.dataframe(df)

        # Extract Brand
        df["Brand"] = df["name"].str.split().str[0]

        df = pd.get_dummies(
            df,
            columns=["fuel", "seller_type"],
            drop_first=True,
            dtype=int
        )

        df = pd.get_dummies(
            df,
            columns=["Brand"],
            drop_first=True,
            dtype=int
        )

        # Remove unused columns
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

        if "seats" in df.columns:
            df.drop(columns=["seats"], inplace=True)

        # -------------------------
        # Match Training Features
        # -------------------------

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # -------------------------
        # Scaling
        # -------------------------

        X = scaler.transform(df)

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(X)

        result = pd.DataFrame({
            "Predicted Price": prediction
        })

        st.subheader("Prediction")

        st.dataframe(result)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction",
            csv,
            "Prediction.csv",
            "text/csv"
        )
