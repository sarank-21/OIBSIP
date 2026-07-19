# 📧 Email Spam Detection with Machine Learning

## 📌 About the Project

This project is an end-to-end **NLP binary classification system** that separates spam SMS/email messages from legitimate ("ham") ones. Raw text is cleaned with a **signal-preserving** preprocessing pipeline, converted into numeric features using **TF-IDF** (unigrams + bigrams), and fed into three classifiers — **Multinomial Naive Bayes**, **Logistic Regression**, and a **calibrated Linear SVM** — which are compared on accuracy, precision, recall, and F1-score. The SVM's decision threshold is then tuned to improve recall without sacrificing much precision, and the tuned model is wrapped in an interactive **Streamlit** app for real-time single-message and batch CSV predictions.

---

## 🛠️ Development Process

1. **Data Collection & Cleaning**
   - Loaded the raw `spam.csv` dataset (5,572 SMS messages, `latin1` encoding).
   - Dropped three empty artifact columns (`Unnamed: 2/3/4`) left over from the CSV export.
   - Renamed `v1` → `target` and `v2` → `message` for clarity.
   - Checked and removed duplicate rows with `drop_duplicates()`.

2. **Text Preprocessing (Signal-Preserving)**
   - Built a `preprocess_text()` pipeline that, instead of deleting URLs, emails, currency symbols, and numbers, replaces them with placeholder tokens (`URL`, `EMAILADDR`, `CURR`, `NUM`) so this signal survives into the feature set.
   - Flags runs of excess punctuation (`!!`, `??`) with `multiexclaim` / `multiquestion` tokens.
   - Lowercases, strips remaining non-alphabetic characters, tokenizes, removes NLTK English stopwords, and applies **Porter Stemmer** stemming to everything except the placeholder tokens.
   - Stored the result in a new `clean_message` column, used for all downstream feature extraction.

3. **Feature Engineering (EDA support)**
   - Derived `num_characters`, `num_words`, and `num_sentences` per message to quantify how spam and ham differ structurally.

4. **Exploratory Data Analysis**
   - Visualized class distribution (found the dataset is imbalanced: **~87% ham vs. ~13% spam**).
   - Compared character/word/sentence-length distributions between spam and ham with histograms and a boxplot.
   - Built a correlation heatmap of the length features.
   - Generated **WordClouds** for spam and ham vocabularies to visually confirm which words drive each class (`free`, `win`, `claim`, `mobile`, `prize` for spam vs. everyday conversational words for ham).

5. **Feature Extraction — TF-IDF**
   - Vectorized `clean_message` with `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`, capturing both single words and word pairs, capped at the 3,000 most informative terms.
   - Encoded the target as `ham → 0`, `spam → 1`.

6. **Train / Test Split**
   - Split the TF-IDF matrix 80/20 with `train_test_split(..., stratify=y, random_state=42)` to preserve the ham/spam ratio in both sets.
   - Added `class_weight='balanced'` to Logistic Regression and the SVM so the models stop leaning toward the majority (ham) class.
   - Wrapped `LinearSVC` in `CalibratedClassifierCV` (cv=3) to add `predict_proba` support, since plain `LinearSVC` only gives hard 0/1 predictions — calibration is what enables the threshold-tuning step later.

7. **Model Building**
   - Trained and evaluated three classifiers through a shared `evaluate_model()` helper: **Multinomial Naive Bayes**, **Logistic Regression** (`max_iter=1000, class_weight='balanced'`), and **calibrated Linear SVM** (`class_weight='balanced', max_iter=5000`).

8. **Model Evaluation**
   - Compared all three models on Accuracy, Precision, Recall, and F1-score.
   - Plotted a grouped bar chart of the four metrics per model.
   - Rendered confusion matrices and full `classification_report`s side by side.

9. **Decision Threshold Tuning**
   - Swept the SVM's spam-decision threshold (0.5 down to 0.25) and tracked Recall, Precision, F1, and Accuracy at each cutoff.
   - Plotted the recall/precision/F1 trade-off curve and selected the threshold with the best F1 (~0.45) as the deployed default — recovering several spam messages that sat just under the old 0.5 cutoff at only a small precision cost.

10. **Discussion**
    - Discussed why recall is particularly important for spam detection, while noting production filters still keep precision high, since a missed spam email is a minor annoyance but a legitimate email wrongly flagged as spam can be costly.
    - Noted Naive Bayes and Logistic Regression have very high precision, while the tuned Linear SVM achieves the best recall/F1 balance.

11. **Model Selection & Persistence**
    - Deployed the threshold-tuned Linear SVM ("Linear SVM (tuned)"), rather than picking purely by default-threshold F1.
    - Serialized the best model, the fitted TF-IDF vectorizer, the model name, its tuned metrics, and the chosen decision threshold to disk with `joblib`.

12. **Dashboard Development**
    - Wrote a Streamlit app (`streamlit_app_3.py`) that loads the saved artifacts and re-applies the exact same `preprocess_text()` pipeline before predicting.
    - Serves two modes via a radio toggle: **Single Email** (text area → spam/ham verdict with a confidence progress bar) and **CSV Upload** (bulk prediction with a downloadable results CSV).
    - Launched the app directly from the notebook as a background process.

---

## 🔑 Key Features

### 🧹 Signal-Preserving Text Cleaning
Replaces URLs, emails, currency symbols, numbers, and excess punctuation with placeholder tokens instead of deleting them, then lowercases, removes stopwords, and stems — keeping spam-indicative signal that naive cleaning would throw away.

### 📊 Class Imbalance Handling
Measures and visualizes the ~87/13 ham/spam split, then actively compensates for it with `class_weight='balanced'` on Logistic Regression and the SVM.

### ☁️ Spam vs. Ham WordClouds
Side-by-side WordCloud visualization exposing the vocabulary that most separates spam from ham.

### 🔢 TF-IDF with Bigrams
Converts cleaned text into a 3,000-dimension TF-IDF matrix using both unigrams and bigrams (`ngram_range=(1, 2)`) for richer phrase-level signal.

### 🤖 Multi-Model Benchmarking
Trains and fairly compares Naive Bayes, Logistic Regression, and a calibrated Linear SVM on identical train/test splits.

### 🎚️ Decision Threshold Tuning
Sweeps the SVM's spam-probability cutoff and picks the threshold that best balances recall against precision, rather than accepting the default 0.5.

### 📈 Rich Evaluation Suite
Metrics table, comparison bar chart, confusion matrices, full classification reports, and a threshold trade-off curve.

### 🧠 Automated Best-Model Selection
Deploys the threshold-tuned Linear SVM and persists it (plus the vectorizer and threshold) with `joblib` — no manual step required.

### 🖥️ Streamlit Prediction App
A two-mode UI: paste a single email for an instant spam/ham verdict with a live confidence bar, or upload a CSV for batch predictions with a downloadable results file.

### 📁 Batch CSV Prediction
Upload any CSV with an `email` column and get a spam/ham label plus a spam-confidence percentage appended to every row, downloadable as a new CSV.

---

## 🔍 Features (Detailed)

**Data Cleaning & Preprocessing**
- Drops the three empty trailing columns from the raw Kaggle-style `spam.csv` export.
- Renames columns to `target` / `message` for readability.
- Removes exact duplicate rows before any modeling begins.
- Signal-preserving `preprocess_text()` function applied consistently across the dataset and reused verbatim inside the Streamlit app, so training-time and inference-time cleaning never drift apart.

**Exploratory Data Analysis**
- Class distribution table and count plot.
- Three-panel histogram (characters/words/sentences) split by class, plus a boxplot and a correlation heatmap.
- WordCloud visualization pair (spam vs. ham) for qualitative vocabulary inspection.

**Modeling & Evaluation**
- Shared `evaluate_model()` utility keeps training/evaluation logic identical across all three algorithms.
- Metrics comparison table sorted by F1-score.
- Grouped bar chart of Accuracy / Precision / Recall / F1 across models.
- Confusion matrices rendered as heatmaps for all three models side by side.
- Full `classification_report` per model.
- Threshold-sweep table and trade-off plot (Recall/Precision/F1 vs. threshold) for the calibrated SVM, plus a written discussion of why recall and precision trade off differently in a spam-filtering context.

**Deployment**
- Best (threshold-tuned) model, vectorizer, model name, tuned metrics, and decision threshold saved via `joblib`.
- Streamlit app with a radio toggle for Single Email vs. CSV Upload, a spam-confidence progress bar for single predictions, and a downloadable results CSV for batch predictions.

---

## 🧰 Tech Stack

**🧠 Machine Learning**
- `scikit-learn` — `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `CalibratedClassifierCV`, `train_test_split`, and the full metrics suite (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`)

**📊 Data Processing & Analysis**
- `pandas` — dataset loading, cleaning, and manipulation
- `numpy` — numerical operations

**📝 Natural Language Processing**
- `nltk` — stopwords, `sent_tokenize`, `PorterStemmer`
- `re` / `string` — regex-based text cleaning (URL/email/currency/number/punctuation detection)

**📈 Data Visualization**
- `matplotlib` — plotting, including the threshold trade-off curve
- `seaborn` — count plots, histograms, boxplots, heatmaps
- `wordcloud` — spam/ham WordClouds

**🖥️ Application / Deployment**
- `streamlit` — interactive prediction UI (single email + batch CSV, with confidence display and CSV download)
- `joblib` — model, vectorizer, and threshold persistence

---

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/email-spam-detection.git
   cd email-spam-detection
   ```

2. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `wordcloud`, `streamlit`, `joblib`

4. **Download NLTK Resources** (handled automatically in the notebook and the app, but can be run manually)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('wordnet')
   ```

5. **Prepare the Dataset**
   - Place `spam.csv` in the project directory and update `DATA_PATH` in the notebook to point to it.

6. **Run the Notebook**
   - Open `Task_4-Email_Spam_Detection.ipynb` and run all cells top to bottom. This trains the models, tunes the decision threshold, and saves `best_model.pkl`, `tfidf_vectorizer.pkl`, `best_model_name.pkl`, `metrics.pkl`, and `decision_threshold.pkl`.

7. **Launch the Streamlit App**
   ```bash
   streamlit run streamlit_app_3.py --server.port 8502
   ```
---

## 🗺️ How It Works

```
                 spam.csv (raw SMS dataset)
                          │
                          ▼
              Data Cleaning & Deduplication
        (drop Unnamed cols, rename, drop_duplicates)
                          │
                          ▼
         preprocess_text()  →  clean_message
  (URL/email/currency/number → placeholder tokens
    → lowercase → flag excess punctuation → tokenize
        → remove stopwords → Porter stem)
                          │
                          ▼
     TfidfVectorizer(max_features=3000, ngram_range=(1,2))
                          │
                          ▼
    train_test_split(stratify=y, test_size=0.2, random_state=42)
                          │
          ┌───────────────┼─────────────────────┐
          ▼                ▼                     ▼
   MultinomialNB   LogisticRegression      CalibratedClassifierCV
   (baseline)      (class_weight=balanced)  (LinearSVC, class_weight=balanced)
          │                │                     │
          └───────────────┼─────────────────────┘
                          ▼
              evaluate_model() → metrics
   (Accuracy, Precision, Recall, F1, Confusion Matrix)
                          │
                          ▼
        Threshold sweep on calibrated SVM (0.5 → 0.25)
          → best-F1 threshold chosen (~0.45)
                          │
                          ▼
     joblib.dump → best_model.pkl / tfidf_vectorizer.pkl
                 / decision_threshold.pkl / metrics.pkl
                          │
                          ▼
              streamlit_app_3.py (Streamlit UI)
        ┌─────────────────────┬─────────────────────┐
        ▼                                           ▼
   Single Email Mode                         CSV Upload Mode
 (text_area → predict + confidence bar)  (file_uploader → batch predict
                                            → downloadable results CSV)
```

---

## 📝 Project Overview

This project implements a complete NLP pipeline for spam detection on the well-known SMS Spam Collection dataset (5,572 messages, ~87% ham / ~13% spam). Text is normalized through a signal-preserving cleaning pipeline — replacing URLs, emails, currency, numbers, and excess punctuation with placeholder tokens rather than deleting them — then stopword-removed and Porter-stemmed before being transformed into a 3,000-feature TF-IDF matrix over unigrams and bigrams. Three classifiers — Multinomial Naive Bayes, Logistic Regression, and a calibrated Linear SVM (both class-weighted for the imbalance) — are trained on a stratified 80/20 split and compared on accuracy, precision, recall, and F1-score. The SVM's decision threshold is then swept and tuned to recover extra recall at minimal precision cost, and the resulting **Linear SVM (tuned)** model is serialized with its vectorizer and threshold, then deployed through a Streamlit app supporting both single-email prediction with a live confidence bar and batch CSV prediction with a downloadable results file.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
