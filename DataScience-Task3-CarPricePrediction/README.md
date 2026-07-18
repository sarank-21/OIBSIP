# 🚗 Car Price Prediction with Machine Learning

## 📖 About the Project

This project builds a regression pipeline that predicts the **selling price of a used car** from real-world listing attributes — brand, age, mileage, engine specs, fuel type, transmission, and ownership history. It covers heavy data cleaning (messy text fields like `"1248 CC"` and free-text torque values), feature engineering, categorical encoding, and a three-way model comparison (Linear Regression, Random Forest, XGBoost) before deploying the best model as an interactive Streamlit pricing tool.

---

## 🔨 Development Process

**1. Data Loading**
- Loaded the raw `Car Price Dataset.csv` into a `pandas` DataFrame and inspected shape and null counts

**2. Data Cleaning & Preprocessing**
- Removed exact duplicate rows via `duplicated()` / `drop_duplicates()`
- Audited categorical text columns (`fuel`, `seller_type`, `transmission`, `owner`) for inconsistent values
- Imputed `seats` with the column median
- Parsed unit-suffixed text fields into numeric floats: `engine` (`"1248 CC"` → `1248.0`), `mileage` (`"23.4 kmpl"` → `23.4`), and `max_power` (`"74 bhp"` → `74.0`), each filled with its column median where missing
- Built a custom `extract_torque()` / `extract_rpm()` parser to pull numeric Nm and RPM values out of free-text `torque` strings (e.g. `"190Nm@ 2000rpm"`, `"12.7@ 2,700(kgm@ rpm)"`), converting kgm to Nm where needed, then dropped the original `torque` column
- Detected outliers in `mileage` and `torque_rpm` with boxplots, then clipped values to the IQR bounds (`[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`) instead of dropping rows

**3. Feature Engineering**
- Extracted `Brand` from the first word of the `name` field (e.g. `"Maruti Swift Dzire VDI"` → `"Maruti"`)
- Derived `car_age` from the `year` column relative to the current year, since age is generally more predictive of price than raw manufacture year

**4. Exploratory Data Analysis**
- Visualised the selling-price distribution (right-skewed, long tail toward premium vehicles)
- Compared price against fuel type, transmission, seller type, car age, and kilometers driven with box plots and scatter plots
- Ranked the top 10 brands by average selling price

**5. Encoding**
- Applied an explicit ordinal mapping to `owner` (First → Test Drive Car)
- Binary-mapped `transmission` (Manual/Automatic)
- One-hot encoded low-cardinality nominal features (`fuel`, `seller_type`) with `drop_first=True` to avoid the dummy-variable trap
- One-hot encoded high-cardinality `Brand` separately, after the correlation heatmap, to keep the heatmap readable

**6. Feature Correlation Analysis**
- Generated a correlation heatmap over the numeric/encoded columns (pre-`Brand` encoding) to check for multicollinearity

**7. Train/Test Split & Scaling**
- Split data 80/20 (`random_state=42`), dropping `selling_price` (target), `name` (free text), and `seats` (excluded from the modeled feature set)
- Standardised features with `StandardScaler` — primarily for Linear Regression, kept across all models for a consistent pipeline

**8. Model Training**
- Trained three regressors on the same split: **Linear Regression**, **Random Forest Regressor** (`n_estimators=100`), and **XGBoost Regressor** (`n_estimators=100`, `learning_rate=0.05`, `max_depth=5`)

**9. Model Evaluation**
- Scored every model on **MAE**, **RMSE**, and **R² score**, ranked in a comparison table

**10. Feature Importance & Deployment**
- Automatically selected the best model by R² score, then plotted its top 15 most influential features (`feature_importances_` for tree models, absolute coefficients for Linear Regression)
- Persisted the winning model, scaler, feature schema, and metrics with `joblib`, then wrapped it in a Streamlit app supporting both manual input and CSV upload

---

## 🔎 Key Features

### 🧹 Messy Real-World Data Cleaning
Custom parsers turn unit-suffixed and free-text fields (`engine`, `mileage`, `max_power`, `torque`) into clean numeric columns.

### 📐 IQR-Based Outlier Clipping
Extreme values in `mileage` and `torque_rpm` are clipped to IQR bounds rather than dropped, preserving every record.

### 🏷️ Brand & Age Feature Engineering
Brand is extracted straight from the car name, and car age is derived from manufacture year for stronger predictive signal.

### 📊 Extensive EDA
Price is examined against fuel type, transmission, seller type, age, mileage, and brand through histograms, box plots, and scatter plots.

### 🔢 Feature-Aware Encoding
Ordinal, binary, and one-hot encoding are each applied where they fit the feature's actual structure.

### ⚖️ Three-Model Regression Comparison
Linear Regression, Random Forest, and XGBoost are trained and benchmarked side-by-side on MAE, RMSE, and R².

### 🏆 Automatic Best-Model Selection
The top-performing regressor is chosen programmatically by R² score, no manual picking required.

### 📈 Feature Importance Visualisation
The top 15 price drivers for the winning model are plotted automatically.

### 💾 Full Pipeline Persistence
The trained model, scaler, feature schema, model name, and metrics are all saved with `joblib` for reuse.

### 🖥️ Dual-Mode Streamlit Deployment
The deployed app supports both manual car-detail entry and batch CSV upload, replicating the exact training-time feature encoding at inference.

---

## 🧩 Features (Detailed)

### Data Cleaning Pipeline
- Duplicate removal, missing-value percentage audit, and median imputation for `seats`, `engine`, `mileage`, and `max_power`
- Regex-based `extract_torque()` and `extract_rpm()` functions handling multiple free-text torque formats, including kgm→Nm conversion
- IQR-based clipping applied per-column after boxplot inspection

### Exploratory Analysis
- Selling price distribution histogram with KDE overlay
- Price vs. fuel type, transmission, seller type (box plots)
- Price vs. car age, kilometers driven (scatter plots)
- Top 10 brands by average selling price (bar chart)

### Encoding Strategy
- Ordinal mapping for `owner` (preserves real ownership order)
- Binary mapping for `transmission`
- One-hot encoding (`drop_first=True`) for `fuel`, `seller_type`, and `Brand`, applied in separate stages to keep the correlation heatmap interpretable

### Model Training & Evaluation
- Consistent scaled-feature pipeline across Linear Regression, Random Forest, and XGBoost
- MAE, RMSE, and R² computed for every model and ranked in a results table
- Feature importance plot generated dynamically for whichever model wins

### Deployment
- `streamlit_app_2.py` generated directly from the notebook via `%%writefile`
- Loads `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `best_model_name.pkl`, and `metrics.pkl`
- Manual input mode replicates encoding (owner map, transmission map, brand extraction, one-hot encoding, feature alignment) before scaling and predicting
- CSV upload mode applies the same encoding pipeline in batch and offers a downloadable prediction CSV

---

## 🛠️ Tech Stack

**📊 Data Processing & Analysis**
- `pandas` — DataFrame construction, cleaning, encoding, aggregation
- `numpy` — numeric operations, IQR/outlier calculations
- `re` — regex parsing for torque/RPM extraction
- `datetime` — computing car age from manufacture year

**🧠 Machine Learning**
- `scikit-learn` — `LinearRegression`, `RandomForestRegressor`, `train_test_split`, `StandardScaler`, evaluation metrics
- `xgboost` — `XGBRegressor` for gradient-boosted regression

**📈 Data Visualisation**
- `matplotlib` — histograms, custom plot styling
- `seaborn` — box plots, scatter plots, bar charts, correlation heatmap
- `plotly.express` — interactive boxplots for outlier inspection

**🖥️ Deployment**
- `streamlit` — interactive web app with manual input and CSV upload modes
- `joblib` — model, scaler, feature schema, and metrics persistence

---

## ⚙️ Setup & Installation

**1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction
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
# Core libraries: numpy pandas matplotlib seaborn plotly scikit-learn xgboost joblib streamlit
```

**4. Prepare the Dataset**
Place `Car Price Dataset.csv` in your project directory and update the `DATA_PATH` in the notebook to point to it.

**5. Run the Notebook**
Open the notebook in Jupyter and run all cells in order to clean the data, train the models, and generate `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `best_model_name.pkl`, and `metrics.pkl`.

**6. Run the Streamlit App**
```bash
streamlit run streamlit_app_2.py
```

## 🧭 How It Works

```
              Car Price Dataset.csv → pandas DataFrame
                              │
              Data Cleaning (duplicates, nulls,
        engine/mileage/max_power parsing, torque
           extraction via extract_torque/extract_rpm,
                  IQR-based outlier clipping)
                              │
             Feature Engineering (Brand extraction,
                     car_age derivation)
                              │
              Exploratory Data Analysis (price vs.
          fuel, transmission, seller_type, age, km_driven,
                       brand comparisons)
                              │
        Encoding (ordinal: owner | binary: transmission |
             one-hot: fuel, seller_type, Brand)
                              │
              Correlation Heatmap → train_test_split()
                          → StandardScaler
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                      │
 LinearRegression   RandomForestRegressor      XGBRegressor
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              │
           Evaluation (MAE, RMSE, R² comparison table)
                              │
          Best Model Selection (highest R²) →
             Feature Importance Plot (top 15)
                              │
       joblib.dump → best_model.pkl / scaler.pkl /
     feature_columns.pkl / best_model_name.pkl / metrics.pkl
                              │
                  streamlit_app_2.py
       (Manual Input  or  Upload CSV → same encoding
                pipeline → prediction)
                              │
              Predicted Selling Price Output
```

---

## 📋 Project Overview

This project is a supervised regression pipeline built to predict used-car selling prices from a messy, real-world listings dataset, comparing three algorithms — Linear Regression, Random Forest, and XGBoost — to find the strongest predictor of price. The core technical approach centers on heavy data cleaning: parsing unit-suffixed and free-text fields like engine displacement, mileage, and torque into usable numeric features, engineering `Brand` and `car_age` from raw listing text, and applying encoding strategies (ordinal, binary, one-hot) matched to each categorical feature's structure. Models are evaluated with MAE, RMSE, and R², and the best-performing one is selected automatically and interrogated for feature importance — consistently surfacing car age and engine power as top price drivers. The winning model, scaler, and feature schema are persisted with `joblib` and served through a Streamlit app that replicates the exact training-time encoding pipeline for both manual entry and batch CSV predictions. The result is a realistic, end-to-end regression project that handles the kind of inconsistent, unit-laden data typical of real automotive listings rather than a pre-cleaned toy dataset.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
