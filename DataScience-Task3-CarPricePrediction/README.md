# 🚗 Car Price Prediction with Machine Learning

## Objective
Build a regression model that predicts the **selling price of a used car** based on features such as brand, age, mileage, fuel type, engine specs, and transmission.

## Technologies Used
- **Python** — core language
- **pandas / numpy** — data loading, cleaning, and manipulation
- **scikit-learn** — preprocessing (`StandardScaler`), Linear Regression, Random Forest Regressor, evaluation metrics
- **xgboost** — XGBoost Regressor
- **matplotlib / seaborn / plotly** — visualization and EDA

## Dataset Information
- **Source file:** `Car Price Dataset.csv`
- **Size:** 8,128 rows × 13 columns loaded; **6,926 rows** after removing duplicates
- **Target variable:** `selling_price`

| Column | Description |
|---|---|
| `name` | Car model name (includes brand) |
| `year` | Year of manufacture |
| `selling_price` | Selling price of the car (**target**) |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type (Petrol/Diesel/CNG/LPG/Electric) |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | Ownership history (First/Second/Third owner, etc.) |
| `mileage` | Fuel efficiency (kmpl or km/kg) |
| `engine` | Engine displacement (CC) |
| `max_power` | Maximum power output (bhp) |
| `torque` | Torque details (free text — parsed into `torque_nm` / `torque_rpm`) |
| `seats` | Number of seats |

Missing values (~3% each in `mileage`, `engine`, `max_power`, `torque`, `seats`) were imputed with the median/mode after cleaning.

## Steps / Workflow
1. **Data Loading** — read the raw CSV (8,128 rows × 13 columns)
2. **Data Cleaning**
   - Removed duplicate rows (8,128 → 6,926)
   - Standardized inconsistent categorical text (`fuel`, `seller_type`, `transmission`, `owner`)
   - Handled missing values:
     - `seats` → filled with median
     - `engine` (e.g. `"1248 CC"`) → numeric extraction, filled with median
     - `mileage` (e.g. `"23.4 kmpl"`) → numeric extraction, filled with median
     - `max_power` (e.g. `"74 bhp"`) → numeric extraction, filled with median
     - `torque` (free text, e.g. `"190Nm@ 2000rpm"`) → parsed into `torque_nm` and `torque_rpm`, with kgm→Nm conversion where needed, filled with median
   - Outlier handling: IQR-based clipping on `mileage` and `torque_rpm`
3. **Feature Engineering**
   - `Brand` — extracted from the first word of `name`
   - `car_age` — computed as `current_year - year`
4. **Exploratory Data Analysis (EDA)**
   - Selling price distribution (right-skewed)
   - Selling price vs. fuel type, transmission, seller type (boxplots)
   - Selling price vs. car age, km driven (scatterplots — negative correlation with both)
   - Top 10 brands by average selling price
5. **Encoding Categorical Features**
   - `owner` → ordinal mapping (First → Test Drive Car)
   - `transmission` → binary mapping (Manual/Automatic)
   - `fuel`, `seller_type` → one-hot encoding (`drop_first=True`)
   - `Brand` → one-hot encoding (applied after the correlation heatmap, due to high cardinality)
6. **Feature Correlation Heatmap** — computed on numeric/encoded columns
7. **Train/Test Split & Scaling** — 80/20 split, features standardized with `StandardScaler`
8. **Model Training** — Linear Regression, Random Forest Regressor, XGBoost Regressor
9. **Model Evaluation** — MAE, RMSE, R² score for each model
10. **Feature Importance** — top 15 features for the best-performing model

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Random Forest Regressor** | **72,438.22** | **123,660.48** | **0.9303** |
| XGBoost Regressor | 76,982.22 | 129,977.58 | 0.9230 |
| Linear Regression | 133,010.59 | 256,088.45 | 0.7010 |

*(Lower MAE/RMSE is better; R² closer to 1.0 is better. Prices in the dataset's native currency units.)*

## Results
- **Random Forest Regressor** was the best-performing model, explaining **~93% of the variance** in selling price (R² = 0.9303) with the lowest average error.
- **XGBoost Regressor** performed almost as well (R² = 0.9230), close behind Random Forest.
- **Linear Regression** lagged significantly (R² = 0.7010), confirming that car pricing depends on **non-linear interactions** between features (e.g. the effect of age on price differs by brand and fuel type) — something tree-based ensemble models capture naturally but a linear model cannot.
- **Feature importance** (from Random Forest) shows `car_age` and `max_power` as the strongest predictors of selling price — consistent with intuition that newer, more powerful cars from premium brands command higher resale value.

## How to Run the Project

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
```

### 2. Set the dataset path
Update the `DATA_PATH` variable in the notebook to point to your local copy of the dataset:
```python
from pathlib import Path
DATA_PATH = Path("path/to/Car Price Dataset.csv")
```

### 3. Run the notebook
Open `Task_3_-_Car_Price_Prediction_with_Machine_Learning.ipynb` in Jupyter Notebook / JupyterLab / VS Code and run all cells top to bottom:
```bash
jupyter notebook "Task_3_-_Car_Price_Prediction_with_Machine_Learning.ipynb"
```

The notebook will:
- Clean and impute the raw dataset
- Engineer `Brand` and `car_age` features
- Generate EDA plots (distributions, boxplots, scatterplots, correlation heatmap)
- Encode categorical features and scale numeric ones
- Train and evaluate Linear Regression, Random Forest, and XGBoost
- Automatically select the best model and plot its top 15 feature importances
