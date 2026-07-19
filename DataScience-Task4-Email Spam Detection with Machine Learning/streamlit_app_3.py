
import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
model_name = joblib.load("best_model_name.pkl")
metrics = joblib.load("metrics.pkl")

st.set_page_config(page_title="Email Spam Detection", layout="wide")

st.title("📧 Email Spam Detection")

st.success(f"Best Model : {model_name}")

st.subheader("Model Performance")

metric_df = pd.DataFrame(metrics.items(), columns=["Metric","Value"])
st.table(metric_df)

tab1, tab2 = st.tabs(["Single Prediction","CSV Prediction"])

# ----------------------------
# Single Prediction
# ----------------------------
with tab1:

    st.subheader("Predict Single Email")

    email = st.text_area("Enter Email Text")

    if st.button("Predict"):

        if email.strip() == "":
            st.warning("Please enter email text.")
        else:

            vector = tfidf.transform([email])

            prediction = model.predict(vector)[0]

            if prediction == 1:
                st.error("🚨 Spam Email")
            else:
                st.success("✅ Ham (Not Spam)")


# ----------------------------
# CSV Prediction
# ----------------------------
with tab2:

    st.subheader("Predict Multiple Emails")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(df.head())

        # Change this column name if needed
        TEXT_COLUMN = "text"

        if TEXT_COLUMN not in df.columns:

            st.error(f"CSV must contain '{TEXT_COLUMN}' column.")

        else:

            X = tfidf.transform(df[TEXT_COLUMN].astype(str))

            pred = model.predict(X)

            df["Prediction"] = [
                "Spam" if i==1 else "Ham"
                for i in pred
            ]

            st.success("Prediction Completed")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )
