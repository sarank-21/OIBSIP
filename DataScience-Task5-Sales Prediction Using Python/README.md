# 📈 Sales Prediction Using Python

## 📌 About the Project

This project builds a **regression pipeline** that predicts product sales from advertising spend across three media channels — **TV**, **Radio**, and **Newspaper**. After exploring how each channel relates to sales, a **Linear Regression** baseline and a **Random Forest Regressor** are trained and compared on MAE, RMSE, and R², with residual analysis used to sanity-check the winning model. The pipeline also interprets which channel drives sales most, using both linear coefficients and Random Forest feature importances, and ships the best model as an interactive **Streamlit** app for manual and batch CSV predictions.

---

## 🛠️ Development Process

1. **Data Loading**
   - Loaded `Advertising.csv` (TV, Radio, Newspaper spend and Sales).
   - Renamed the unlabeled index column `Unnamed: 0` → `Sales_id`.

2. **Data Quality Checks**
   - Inspected structure and dtypes with `.info()`.
   - Checked for missing values (`isnull().sum()`) and duplicate rows (`duplicated().sum()`).
   - Reviewed summary statistics with `.describe()` to spot scale/skew before modeling.

3. **Exploratory Data Analysis**
   - Built a full pairplot of TV, Radio, Newspaper, and Sales to scan every pairwise relationship at once.
   - Plotted individual scatter plots of Sales vs. each channel (TV, Radio, Newspaper) to read relationship strength and shape.
   - Computed a Pearson correlation matrix and visualized it as a heatmap — confirming TV has the strongest correlation with Sales, followed by Radio, with Newspaper weakest.

4. **Train / Test Split**
   - Split features (`TV`, `Radio`, `Newspaper`) from the target (`Sales`) with an 80/20 `train_test_split(random_state=42)`.

5. **Feature Scaling**
   - Standardized features with `StandardScaler` (fit on train only) so Linear Regression coefficients are directly comparable across channels.

6. **Model Training**
   - Trained a **Linear Regression** baseline on the scaled features.
   - Trained a **Random Forest Regressor** (`n_estimators=100, random_state=42`) on the raw (unscaled) features, since tree ensembles don't require scaling.
   - Both models run through a shared `evaluate_model()` helper that fits, predicts, and computes MAE, MSE, RMSE, and R².

7. **Model Evaluation & Comparison**
   - Collected both models' metrics into a single results table sorted by R².
   - Plotted a grouped bar chart of MAE, RMSE, and R² across models — Random Forest came out ahead on all three.

8. **Residual Analysis**
   - Computed residuals (actual − predicted) for the best model (Random Forest).
   - Plotted residuals vs. predicted Sales and their distribution to check for systematic error patterns — found residuals randomly scattered around zero.

9. **Feature Importance / Interpretation**
   - Extracted standardized Linear Regression coefficients and Random Forest feature importances.
   - Visualized both side by side, confirming **TV** drives Sales the most, followed by **Radio**, with **Newspaper** contributing the least.

10. **Model Selection & Persistence**
    - Programmatically re-selected the best model by highest R² score.
    - Saved the model, `StandardScaler`, a `needs_scaling` flag, feature column order, model name, metrics, and per-feature numeric ranges (for UI sliders) with `joblib`.

11. **Dashboard Development**
    - Wrote a Streamlit app (`streamlit_app_4.py`) that loads all saved artifacts and offers **Manual Input** and **Upload CSV** prediction modes, applying scaling only when the loaded model requires it.
    - Launched the app directly from the notebook as a background process.

---

## 🔑 Key Features

### 🔎 Full Exploratory Analysis
Pairplot, per-channel scatter plots, and a correlation heatmap give a complete picture of how each ad channel relates to Sales before any modeling.

### 📊 Two-Model Benchmarking
Trains and fairly compares an interpretable Linear Regression baseline against a non-linear Random Forest Regressor on identical train/test splits.

### 📐 Standardized, Comparable Coefficients
Features are scaled before Linear Regression so coefficient magnitudes can be directly compared across TV, Radio, and Newspaper.

### 📈 Full Regression Metric Suite
Reports MAE, MSE, RMSE, and R² for every model, plus a comparison bar chart.

### 🩺 Residual Diagnostics
Residual-vs-predicted and residual-distribution plots sanity-check whether the best model's errors are random noise or a systematic miss.

### 🌲 Dual Feature-Importance View
Compares Linear Regression coefficients against Random Forest feature importances to triangulate which channel actually drives Sales.

### 🧠 Automated Best-Model Selection
Re-selects the top model by R² and persists it — along with the scaler and a scaling flag — via `joblib`, so the app always deploys the right model correctly.

### 🖥️ Streamlit Prediction App
Supports both **Manual Input** (number inputs seeded from real training ranges) and **Upload CSV** batch prediction modes.

### 📁 Batch CSV Prediction
Validates uploaded CSV columns against the training feature set and appends a `Prediction` column for every row.

---

## 🔍 Features (Detailed)

**Data Quality & EDA**
- Missing-value and duplicate-row checks before any transformation.
- Descriptive statistics (`.describe()`) to catch scale differences between channels.
- Pairplot plus per-channel scatter plots plus a correlation heatmap for a layered view of the TV/Radio/Newspaper → Sales relationship.

**Modeling & Evaluation**
- Shared `evaluate_model()` utility keeps training/scoring logic identical for both algorithms.
- Metrics comparison table sorted by R² Score.
- Grouped bar chart of MAE, RMSE, and R² across models.
- Residual-vs-predicted scatter plot and residual histogram for the winning model.

**Interpretation**
- Standardized Linear Regression coefficient table and bar chart.
- Random Forest feature-importance table and bar chart, plotted side by side with the coefficients for a two-method comparison.

**Deployment**
- Best model, scaler, `needs_scaling` flag, feature column order, model name, metrics, and numeric feature ranges all saved via `joblib`.
- Streamlit app with a radio toggle between Manual Input and Upload CSV, launched directly from the notebook.

---

## 🧰 Tech Stack

**🧠 Machine Learning**
- `scikit-learn` — `LinearRegression`, `RandomForestRegressor`, `StandardScaler`, `train_test_split`, and regression metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`)

**📊 Data Processing & Analysis**
- `pandas` — dataset loading, cleaning, and manipulation
- `numpy` — numerical operations (e.g. RMSE computation)

**📈 Data Visualization**
- `matplotlib` — plotting
- `seaborn` — pairplot, scatter plots, heatmap, bar plots, residual histogram

**🖥️ Application / Deployment**
- `streamlit` — interactive prediction UI (manual + batch CSV)
- `joblib` — model, scaler, and metadata persistence

---

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/sales-prediction-python.git
   cd sales-prediction-python
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
   Key libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`

4. **Prepare the Dataset**
   - Place `Advertising.csv` in the project directory and update the file path in the notebook to point to it.

5. **Run the Notebook**
   - Open `Task_5-Sales_Prediction_Using_Python.ipynb` and run all cells top to bottom. This trains both models and saves `best_model.pkl`, `scaler.pkl`, `needs_scaling.pkl`, `feature_columns.pkl`, `best_model_name.pkl`, `metrics.pkl`, and `numeric_ranges.pkl`.

6. **Launch the Streamlit App**
   ```bash
   streamlit run streamlit_app_4.py --server.port 8502
   ```

## 🗺️ How It Works

```
              Advertising.csv (TV, Radio, Newspaper, Sales)
                          │
                          ▼
              Rename Unnamed: 0 → Sales_id
                          │
                          ▼
        Data Quality Checks (nulls, duplicates, describe)
                          │
                          ▼
   EDA: pairplot → per-channel scatter plots → correlation heatmap
                          │
                          ▼
      train_test_split(X, y, test_size=0.2, random_state=42)
                          │
                          ▼
              StandardScaler (fit on train only)
                          │
          ┌───────────────┴────────────────┐
          ▼                                 ▼
   LinearRegression                RandomForestRegressor
   (scaled features)                 (raw features)
          │                                 │
          └───────────────┬────────────────┘
                          ▼
              evaluate_model() → metrics
        (MAE, MSE, RMSE, R2 Score) + comparison chart
                          │
                          ▼
       Best model selected by highest R2 Score
                          │
                          ▼
              Residual analysis (best model)
                          │
                          ▼
   Coefficient / Feature-Importance interpretation
                          │
                          ▼
   joblib.dump → best_model.pkl / scaler.pkl / needs_scaling.pkl
                 / feature_columns.pkl / numeric_ranges.pkl
                          │
                          ▼
              streamlit_app_4.py (Streamlit UI)
        ┌─────────────────────┬─────────────────────┐
        ▼                                           ▼
   Manual Input Tab                          Upload CSV Tab
 (number_input per feature → predict)   (file_uploader → batch predict)
```

---

## 📝 Project Overview

This project implements a complete regression workflow for predicting product Sales from advertising spend across TV, Radio, and Newspaper channels, using the classic Advertising dataset. After thorough EDA — pairplots, per-channel scatter plots, and a correlation heatmap — features are standardized and split 80/20 for training and testing. A Linear Regression baseline and a Random Forest Regressor are trained and compared on MAE, RMSE, and R², with the Random Forest coming out ahead; its residuals are then checked for systematic patterns and found to be randomly distributed around zero. Both the standardized Linear Regression coefficients and the Random Forest feature importances agree that TV spend has by far the largest impact on Sales, followed by Radio, with Newspaper contributing comparatively little. The winning model, its scaler, and supporting metadata are serialized with joblib and deployed through a Streamlit app that supports both manual single-scenario predictions and batch CSV forecasting — turning the notebook's analysis into a tool marketers could actually use to plan spend.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
