
import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------
# Download NLTK resources
# ---------------------------------------------------
nltk.download("stopwords")

# ---------------------------------------------------
# Load Saved Model
# ---------------------------------------------------
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
DECISION_THRESHOLD = joblib.load("decision_threshold.pkl")

# Load model information
MODEL_NAME = joblib.load("best_model_name.pkl")
MODEL_METRICS = joblib.load("metrics.pkl")

# ---------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

url_re = re.compile(r"(https?://\S+|www\.\S+)")
email_re = re.compile(r"\S+@\S+")
currency_re = re.compile(r"[₹$€£]")
number_re = re.compile(r"\d+")

KEEP_TOKENS = {
    "url",
    "emailaddr",
    "curr",
    "num",
    "multiexclaim",
    "multiquestion"
}

def preprocess_text(text):
    text = str(text)

    text = url_re.sub(" URL ", text)
    text = email_re.sub(" EMAILADDR ", text)
    text = currency_re.sub(" CURR ", text)

    text = text.lower()

    if re.search(r"!{2,}", text):
        text += " multiexclaim "

    if re.search(r"\?{2,}", text):
        text += " multiquestion "

    text = number_re.sub(" NUM ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        w if w in KEEP_TOKENS else stemmer.stem(w)
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict_with_threshold(vector):

    probabilities = model.predict_proba(vector)[:, 1]

    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    return predictions, probabilities


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Email Spam Detection System")

st.write("### Best Model")
st.success(MODEL_NAME)

st.write("### Model Performance")
st.write(MODEL_METRICS)

choice = st.radio(
    "Choose Prediction Type",
    ["Single Email", "CSV Upload"]
)

# =====================================================
# SINGLE EMAIL
# =====================================================
if choice == "Single Email":

    st.subheader("Single Email Prediction")

    email = st.text_area(
        "Enter Email Text",
        height=200
    )

    if st.button("Predict"):

        if email.strip() == "":
            st.warning("Please enter an email.")

        else:

            clean_email = preprocess_text(email)

            vector = tfidf.transform([clean_email])

            prediction, probability = predict_with_threshold(vector)

            if prediction[0] == 1:

                st.error("🚨 SPAM EMAIL")

            else:

                st.success("✅ HAM EMAIL")

            st.write("### Spam Confidence")

            st.progress(float(probability[0]))

            st.write(f"{probability[0]*100:.2f}%")

# =====================================================
# CSV Upload
# =====================================================
else:

    st.subheader("Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        TEXT_COLUMN = "email"

        if TEXT_COLUMN not in df.columns:

            st.error("CSV must contain 'email' column.")

        else:

            clean_text = df[TEXT_COLUMN].astype(str).apply(preprocess_text)

            X = tfidf.transform(clean_text)

            predictions, probabilities = predict_with_threshold(X)

            df["Prediction"] = predictions

            df["Prediction"] = df["Prediction"].map({
                0: "Ham",
                1: "Spam"
            })

            df["Spam Confidence"] = (probabilities * 100).round(2)

            st.success("Prediction Completed")

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="spam_predictions.csv",
                mime="text/csv"
            )
