# 📧 Email Spam Detection with Machine Learning

## 📌 About the Project

This project is an end-to-end **NLP binary classification system** that separates spam SMS/email messages from legitimate ("ham") ones. Raw text is cleaned and normalized, converted into numeric features using **TF-IDF**, and fed into three different classifiers — **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVM** — which are compared head-to-head on accuracy, precision, recall, and F1-score. The best-performing model is then wrapped in an interactive **Streamlit** app for real-time single-message and batch CSV predictions, making the classifier usable outside the notebook.

---

## 🛠️ Development Process

1. **Data Collection & Cleaning**
   - Loaded the raw `spam.csv` dataset (5,572 SMS messages, `latin1` encoding).
   - Dropped three empty artifact columns (`Unnamed: 2/3/4`) left over from the CSV export.
   - Renamed `v1` → `target` and `v2` → `message` for clarity.
   - Checked and removed duplicate rows with `drop_duplicates()`.

2. **Text Preprocessing**
   - Built a `preprocess_text()` pipeline: lowercase conversion → punctuation/digit removal (regex) → whitespace tokenization → stopword removal (NLTK English stopwords) → **Porter Stemmer** stemming.
   - Stored the result in a new `clean_message` column, used for all downstream feature extraction.

3. **Feature Engineering (EDA support)**
   - Derived `num_characters`, `num_words`, and `num_sentences` per message to quantify how spam and ham differ structurally.

4. **Exploratory Data Analysis**
   - Visualized class distribution (found the dataset is imbalanced: **~87% ham vs. ~13% spam**).
   - Compared character/word/sentence-length distributions between spam and ham with histograms and a boxplot.
   - Built a correlation heatmap of the length features.
   - Generated **WordClouds** for spam and ham vocabularies to visually confirm which words drive each class (`free`, `win`, `claim`, `txt` for spam vs. everyday conversational words for ham).

5. **Feature Extraction — TF-IDF**
   - Vectorized `clean_message` with `TfidfVectorizer(max_features=3000)`, capping the vocabulary at the 3,000 most informative terms.
   - Encoded the target as `ham → 0`, `spam → 1`.

6. **Train / Test Split**
   - Split the TF-IDF matrix 80/20 with `train_test_split(..., stratify=y, random_state=42)` to preserve the ham/spam ratio in both sets.

7. **Model Building**
   - Trained and evaluated three classifiers through a shared `evaluate_model()` helper: **Multinomial Naive Bayes**, **Logistic Regression** (`max_iter=1000`), and **Linear SVM**.

8. **Model Evaluation**
   - Compared all three models on Accuracy, Precision, Recall, and F1-score.
   - Plotted a grouped bar chart of the four metrics per model.
   - Rendered confusion matrices and full `classification_report`s side by side.
   - Discussed the precision/recall trade-off specific to spam filtering (why false positives — legit mail marked as spam — are costlier than false negatives).

9. **Model Selection & Persistence**
   - Programmatically selected the model with the highest F1-score.
   - Serialized the best model, the fitted TF-IDF vectorizer, the model name, and its metrics to disk with `joblib`.

10. **Dashboard Development**
    - Wrote a Streamlit app (`streamlit_app_3.py`) that loads the saved artifacts and serves two prediction modes: single-message and batch CSV.
    - Launched the app directly from the notebook as a background process.

---

## 🔑 Key Features

### 🧹 Automated Text-Cleaning Pipeline
Lowercases, strips punctuation/digits, tokenizes, removes stopwords, and stems every message before it ever reaches the model.

### 📊 Class Imbalance Awareness
Explicitly measures and visualizes the ~87/13 ham/spam split and carries that awareness into evaluation (precision/recall/F1 over raw accuracy).

### ☁️ Spam vs. Ham WordClouds
Side-by-side WordCloud visualization exposing the vocabulary that most separates spam from ham.

### 🔢 TF-IDF Feature Extraction
Converts cleaned text into a 3,000-dimension TF-IDF matrix, balancing term frequency against corpus-wide rarity.

### 🤖 Multi-Model Benchmarking
Trains and fairly compares Naive Bayes, Logistic Regression, and Linear SVM on identical train/test splits.

### 📈 Rich Evaluation Suite
Metrics table, comparison bar chart, confusion matrices, and full classification reports for every model.

### 🧠 Automated Best-Model Selection
Picks the top model by F1-score and persists it (plus the vectorizer) with `joblib` — no manual step required.

### 🖥️ Streamlit Prediction App
A two-tab UI: paste a single message for an instant spam/ham verdict, or upload a CSV for batch predictions.

### 📁 Batch CSV Prediction
Upload any CSV with a `text` column and get a spam/ham label appended to every row.

---

## 🔍 Features (Detailed)

**Data Cleaning & Preprocessing**
- Drops the three empty trailing columns from the raw Kaggle-style `spam.csv` export.
- Renames columns to `target` / `message` for readability.
- Removes exact duplicate rows before any modeling begins.
- Full text-normalization function (`preprocess_text`) applied consistently across the dataset.

**Exploratory Data Analysis**
- Class distribution table and count plot.
- Three-panel histogram (characters/words/sentences) split by class, plus a boxplot and a correlation heatmap.
- WordCloud visualization pair (spam vs. ham) for qualitative vocabulary inspection.

**Modeling & Evaluation**
- Shared `evaluate_model()` utility keeps training/evaluation logic identical across all three algorithms.
- Metrics comparison table sorted by F1-score.
- Grouped bar chart of Accuracy / Precision / Recall / F1 across models.
- Confusion matrices rendered as heatmaps for all three models side by side.
- Full `classification_report` per model, plus a written discussion of why recall and precision trade off differently in a spam-filtering context.

**Deployment**
- Best model, vectorizer, model name, and metrics saved via `joblib`.
- Streamlit app with `st.tabs` for Single Prediction vs. CSV Prediction, launched directly from the notebook.

---

## 🧰 Tech Stack

**🧠 Machine Learning**
- `scikit-learn` — `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `train_test_split`, and the full metrics suite (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`)

**📊 Data Processing & Analysis**
- `pandas` — dataset loading, cleaning, and manipulation
- `numpy` — numerical operations

**📝 Natural Language Processing**
- `nltk` — stopwords, `sent_tokenize`, `PorterStemmer`
- `re` / `string` — regex-based text cleaning

**📈 Data Visualization**
- `matplotlib` — plotting
- `seaborn` — count plots, histograms, boxplots, heatmaps
- `wordcloud` — spam/ham WordClouds

**🖥️ Application / Deployment**
- `streamlit` — interactive prediction UI
- `joblib` — model and vectorizer persistence

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

4. **Download NLTK Resources** (handled automatically in the notebook, but can be run manually)
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
   - Open `Task_4-Email_Spam_Detection_with_Machine_Learning.ipynb` and run all cells top to bottom. This trains the models and saves `best_model.pkl`, `tfidf_vectorizer.pkl`, `best_model_name.pkl`, and `metrics.pkl`.

7. **Launch the Streamlit App**
   ```bash
   streamlit run streamlit_app_3.py --server.port 8502
   ```

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
   (lowercase → strip punctuation/digits → tokenize
         → remove stopwords → Porter stem)
                          │
                          ▼
                 TfidfVectorizer(max_features=3000)
                          │
                          ▼
         train_test_split(stratify=y, test_size=0.2)
                          │
          ┌───────────────┼────────────────┐
          ▼                ▼                ▼
   MultinomialNB   LogisticRegression    LinearSVC
          │                │                │
          └───────────────┼────────────────┘
                          ▼
              evaluate_model() → metrics
   (Accuracy, Precision, Recall, F1, Confusion Matrix)
                          │
                          ▼
         Best model selected by highest F1-score
                          │
                          ▼
     joblib.dump → best_model.pkl / tfidf_vectorizer.pkl
                          │
                          ▼
              streamlit_app_3.py (Streamlit UI)
        ┌─────────────────────┬─────────────────────┐
        ▼                                           ▼
  Single Prediction Tab                     CSV Prediction Tab
 (text_area → predict)                (file_uploader → batch predict)
```

---

## 📝 Project Overview

This project implements a complete, classical NLP pipeline for spam detection on the well-known SMS Spam Collection dataset (5,572 messages, ~87% ham / ~13% spam). Text is normalized through lowercasing, punctuation/digit stripping, stopword removal, and Porter stemming, then transformed into a 3,000-feature TF-IDF matrix. Three classifiers — Multinomial Naive Bayes, Logistic Regression, and Linear SVM — are trained on a stratified 80/20 split and compared on accuracy, precision, recall, and F1-score, with confusion matrices and classification reports providing a deeper look at each model's behavior. The notebook explicitly discusses why recall and precision carry different real-world costs in spam filtering, motivating the choice of F1-score as the primary model-selection criterion. The winning model (Linear SVM in this run) is serialized with its TF-IDF vectorizer and deployed through a lightweight Streamlit app supporting both single-message and batch CSV predictions — turning the notebook's analysis into a usable, interactive tool.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
