# 🌸 Iris Flower Classification

## 📖 About the Project

This project trains and compares multiple machine learning classifiers to identify the species of an iris flower — *Setosa*, *Versicolor*, or *Virginica* — from four physical measurements (sepal length/width, petal length/width). The pipeline covers EDA, visual feature-separation analysis, model training/evaluation across four algorithms, and deployment of the best-performing model as an interactive Streamlit prediction app. It's a compact end-to-end demonstration of the classic ML workflow: data → insight → model → deployment.

---

## 🔨 Development Process

**1. Data Collection**
- Loaded the built-in `load_iris()` dataset from scikit-learn (no external download needed)
- Combined `iris.data` and `iris.target` into a single tidy `pandas` DataFrame with a human-readable `species` column mapped from `iris.target_names`

**2. Data Cleaning & Preprocessing**
- Checked dataset shape (150 rows × 6 columns) and column data types
- Detected and dropped duplicate rows with `df.duplicated()` / `df.drop_duplicates()`
- Verified there were no missing/null values via `df.isnull().sum()`
- Confirmed class balance with `value_counts()` — Setosa and Versicolor at 50 samples each, Virginica at 49 after de-duplication

**3. Exploratory Data Analysis & Visualisation**
- Built a `seaborn` pairplot across all four features, colored by species, to visualise class separability
- Plotted per-feature boxplots (2×2 grid) comparing distributions across species
- Generated a correlation heatmap over the four numeric features

**4. Feature Selection Discussion**
- Identified `petal length (cm)` and `petal width (cm)` as the most discriminative features (near-zero overlap between species, ~0.96 correlation with target)
- Identified `sepal width (cm)` as the weakest/least discriminative feature

**5. Train/Test Split & Scaling**
- Performed an 80/20 stratified `train_test_split` (`random_state=42`) to preserve class ratios
- Applied `StandardScaler` to standardise features for scale-sensitive models (Logistic Regression, KNN)

**6. Model Building**
- Trained four classifiers on the split data: **Logistic Regression**, **K-Nearest Neighbours** (`n_neighbors=5`), **Decision Tree** (`random_state=42`), and **Random Forest** (`n_estimators=100`, `random_state=42`)

**7. Model Evaluation**
- Scored every model on Accuracy, Precision, Recall, and F1-score (weighted average) plus a full `classification_report`
- Visualised per-model confusion matrices side-by-side and a bar chart comparing accuracy across all four models

**8. Best Model Selection & Deployment**
- Programmatically selected the top-scoring model from the results table
- Persisted the winning model, scaler, and label metadata with `joblib`, then wrapped it in a Streamlit app for interactive predictions

---

## 🔎 Key Features

### 🧹 Automated Data Cleaning
Duplicate and null checks run automatically before any analysis touches the data.

### 📊 Rich Exploratory Visualisations
Pairplots, per-feature boxplots, and a correlation heatmap reveal how species separate across measurements.

### 🧠 Feature Selection Reasoning
A written discussion (backed by the plots) identifies petal measurements as the most discriminative features.

### 🎯 Multi-Model Comparison
Four classifiers — Logistic Regression, KNN, Decision Tree, Random Forest — are trained and benchmarked side-by-side.

### 📈 Full Evaluation Suite
Accuracy, Precision, Recall, F1-score, classification reports, and confusion matrices for every model.

### 🏆 Automatic Best-Model Selection
The top-performing model is chosen programmatically from the results table, no manual picking required.

### 💾 Model Persistence
The winning model, scaler, and feature/label metadata are saved with `joblib` for reuse outside the notebook.

### 🖥️ One-Click Streamlit Deployment
The saved model is wired directly into a Streamlit app, launched straight from the notebook.

### 📥 Dual Prediction Modes
The deployed app supports both CSV batch upload and manual single-sample input for predictions.

---

## 🧩 Features (Detailed)

### Exploratory Data Analysis
- Shape, dtypes, missing-value, and duplicate checks
- Descriptive statistics via `df.describe()`
- Class balance verification across the three species

### Visual Analysis
- `sns.pairplot` colored by species across all four features
- 2×2 boxplot grid comparing each feature's distribution by species
- Correlation heatmap (`coolwarm` colormap) over the numeric feature set

### Model Training & Evaluation
- Scaled features fed to Logistic Regression and KNN
- Weighted Precision/Recall/F1 computed for imbalanced-safe scoring
- Side-by-side confusion matrix grid (one heatmap per model) for error analysis
- Horizontal bar chart ranking models by test accuracy

### Deployment
- `streamlit_app.py` generated directly from the notebook via `%%writefile`
- Loads `best_model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `target_names.pkl`
- Supports CSV upload for batch predictions or manual numeric input for a single prediction

---

## 🛠️ Tech Stack

**📊 Data Processing & Analysis**
- `pandas` — DataFrame construction, cleaning, aggregation
- `numpy` — numeric array handling
- `scikit-learn` (`datasets`, `model_selection`, `preprocessing`) — dataset loading, train/test split, feature scaling

**🧠 Machine Learning**
- `scikit-learn` — `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`
- `scikit-learn.metrics` — accuracy, precision, recall, F1, confusion matrix, classification report

**📈 Data Visualisation**
- `matplotlib` — plotting engine for boxplots, heatmaps, bar charts
- `seaborn` — pairplots, boxplots, correlation heatmap, confusion matrix heatmaps
- `plotly.express` — imported for interactive visualisation support

**🖥️ Deployment**
- `streamlit` — interactive web app for serving predictions
- `joblib` — model, scaler, and metadata persistence

---

## ⚙️ Setup & Installation

**1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/iris-flower-classification.git
cd iris-flower-classification
```

**2. Create a Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
# Core libraries: numpy pandas matplotlib seaborn plotly scikit-learn joblib streamlit
```

**4. Prepare the Dataset**
No external dataset needed — the Iris data loads directly from `sklearn.datasets.load_iris()`.

**5. Run the Notebook**
Open `Task_1-Iris_Flower_Classification.ipynb` in Jupyter and run all cells in order to train the models and generate `best_model.pkl`, `scaler.pkl`, `feature_names.pkl`, and `target_names.pkl`.

**6. Run the Streamlit App**
```bash
streamlit run streamlit_app.py
```

## 🧭 How It Works

```
                    load_iris() → pandas DataFrame
                              │
                    Data Cleaning & EDA
             (duplicate check, null check, describe())
                              │
              Visualisation Layer (pairplot, boxplots,
                      correlation heatmap)
                              │
              Feature Selection Discussion (petal
                 length/width identified as key)
                              │
           train_test_split()  →  StandardScaler
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                      │
 LogisticRegression   KNeighborsClassifier    DecisionTreeClassifier
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
                    RandomForestClassifier
                              │
          Evaluation (accuracy, precision, recall,
             F1, confusion matrices, bar chart)
                              │
              Best Model Selection (highest
                    test accuracy)
                              │
       joblib.dump → best_model.pkl / scaler.pkl /
         feature_names.pkl / target_names.pkl
                              │
                  streamlit_app.py
          (CSV upload  or  manual number_input)
                              │
                   Predicted Species Output
```

---

## 📋 Project Overview

This project is a supervised multi-class classification pipeline built around the classic Iris dataset, comparing four algorithms — Logistic Regression, K-Nearest Neighbours, Decision Tree, and Random Forest — to identify the strongest performer for distinguishing Setosa, Versicolor, and Virginica flowers. The core technical approach combines exploratory visualisation (pairplots, boxplots, correlation heatmaps) with a data-driven feature selection discussion, followed by a stratified train/test split, feature scaling, and a full evaluation suite covering accuracy, precision, recall, F1-score, and confusion matrices. The best-performing model is selected programmatically and persisted with `joblib` alongside its scaler and label metadata. That saved model powers a lightweight Streamlit app offering both batch CSV predictions and manual single-sample input. The result is a compact but complete demonstration of the ML lifecycle — from raw data to a usable, interactive prediction tool — well suited as a teaching example or a template for other small classification tasks.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
