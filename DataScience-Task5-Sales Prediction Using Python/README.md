# Sales Prediction Using Python

## 📌 Objective
Build a regression model that predicts product **sales** based on advertising spend across three media channels — **TV**, **Radio**, and **Newspaper**. The project also identifies which advertising channel contributes the most to driving sales.

## 🛠️ Technologies Used
- **Python 3.12**
- **pandas**, **NumPy** — data handling
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — modeling & evaluation
  - `train_test_split`, `StandardScaler`
  - `LinearRegression`, `RandomForestRegressor`
  - `mean_absolute_error`, `mean_squared_error`, `r2_score`

## 📊 Dataset Information
- **File:** `Advertising.csv`
- **Records:** 200 rows, 5 columns
- **Features:**
  | Column | Description |
  |---|---|
  | `Sales_id` | Row identifier |
  | `TV` | Advertising spend on TV |
  | `Radio` | Advertising spend on Radio |
  | `Newspaper` | Advertising spend on Newspaper |
  | `Sales` | Target variable — product sales |
- **Data Quality:** No missing values, no duplicate rows.

## 🔄 Steps / Workflow
1. **Import Libraries** — load pandas, numpy, seaborn, matplotlib, and scikit-learn tools.
2. **Load the Dataset** — read `Advertising.csv` and rename the index column.
3. **Exploratory Data Analysis (EDA)**
   - Checked dataset structure, data types, missing values, and duplicates
   - Reviewed descriptive statistics
   - Visualized pairwise relationships with a pairplot
4. **Sales vs. Each Advertising Channel** — scatter plots comparing Sales against TV, Radio, and Newspaper spend.
5. **Correlation Matrix** — computed Pearson correlations and visualized them as a heatmap.
6. **Train/Test Split** — 80/20 split (160 training samples, 40 testing samples, `random_state=42`).
7. **Feature Scaling** — standardized features using `StandardScaler` (fit on training data only).
8. **Model Training**
   - **Linear Regression** (baseline, trained on scaled features)
   - **Random Forest Regressor** (100 estimators, trained on unscaled features)
9. **Model Evaluation** — compared both models using MAE, MSE, RMSE, and R².
10. **Residual Analysis** — examined residuals of the best-performing model for systematic error patterns.
11. **Feature Importance / Interpretation** — compared Linear Regression coefficients vs. Random Forest feature importances to identify the most influential advertising channel.

## 📈 Model Performance

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 1.4608 | 3.1741 | 1.7816 | 0.8994 |
| **Random Forest Regressor** | **0.6203** | **0.5909** | **0.7687** | **0.9813** |

**Best Model:** Random Forest Regressor — lower error and higher R² than the Linear Regression baseline.

## ✅ Results
- **TV advertising spend** has the strongest relationship with Sales (correlation: 0.78), followed by **Radio** (0.58), while **Newspaper** shows the weakest relationship (0.23).
- Both models agree on channel impact ranking:
  - Linear Regression coefficients: TV (3.76) > Radio (2.79) > Newspaper (0.06)
  - Random Forest feature importance: TV (62.5%) > Radio (36.2%) > Newspaper (1.3%)
- Residuals of the Random Forest model are randomly scattered around zero (mean residual ≈ -0.031), indicating no major systematic errors.
- **Conclusion:** Additional TV advertising budget is likely to produce the largest increase in sales, while Newspaper spend contributes comparatively little.

## ▶️ How to Run the Project
1. **Clone/download** this repository and ensure `Advertising.csv` is available (update the file path in the notebook if needed).
2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Task_5_-_Sales_Prediction_Using_Python.ipynb
   ```
4. **Run all cells** in order (Kernel → Restart & Run All) to reproduce the EDA, model training, evaluation, and interpretation results.
