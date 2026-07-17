# 🌸 Iris Flower Classification

## Objective
Train a machine learning classification model to identify the species of an iris flower — **Setosa**, **Versicolor**, or **Virginica** — from its physical measurements (sepal and petal length/width).

## Technologies Used
- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — dataset loading, `StandardScaler`, Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, evaluation metrics
- **matplotlib / seaborn / plotly** — visualization and EDA

## Dataset Information
- **Source:** the classic **Iris dataset**, loaded directly from `sklearn.datasets.load_iris()` (no external file needed)
- **Size:** 150 samples, 4 numeric features (sepal length, sepal width, petal length, petal width — all in cm), 1 duplicate row removed
- **Classes:** 3 species — `setosa` (50), `versicolor` (50), `virginica` (49, after duplicate removal) — a nearly perfectly balanced dataset
- **Missing values:** none

## Steps / Workflow
1. **Load the data** — build a tidy DataFrame from `load_iris()`, mapping numeric species codes to species names
2. **Exploratory Data Analysis (EDA)**
   - Checked shape, dtypes, duplicates (1 found and removed), missing values (none), descriptive statistics, and class balance
3. **Visualizations**
   - Pairplot of all four features colored by species
   - Boxplots of each feature grouped by species
   - Correlation heatmap between features
4. **Feature Selection Discussion**
   - Petal length and petal width are the **most discriminative** features (almost no overlap between species, ~0.96 correlation with each other and with species)
   - Sepal width is the **least discriminative** feature (heavy overlap between Versicolor and Virginica)
5. **Train/Test Split** — 80/20 split, stratified by species (119 training samples, 30 test samples)
6. **Feature Scaling** — `StandardScaler` applied for distance/gradient-based models (Logistic Regression, KNN)
7. **Model Training** — trained and evaluated four classifiers:
   - Logistic Regression
   - K-Nearest Neighbours (k=5)
   - Decision Tree
   - Random Forest (100 trees)
8. **Evaluation** — accuracy, weighted precision/recall/F1, confusion matrices, and classification reports for each model
9. **Best Model Selection** — programmatically selected the model with the highest test accuracy

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| K-Nearest Neighbors | 0.9333 | 0.9444 | 0.9333 | 0.9327 |
| Decision Tree | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Random Forest | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

*(Test set: 30 samples — 10 per species. Metrics are weighted averages across the 3 classes.)*

**Best-performing model:** Logistic Regression — Test accuracy: **0.9333**

## Results
- All four models reached the **same 93.33% test accuracy** on this run (`random_state=42`), reflecting how well-separated the Iris classes are.
- **Setosa** was classified perfectly by every model (linearly separable from the other two species).
- The main source of error across all models was confusion between **Versicolor and Virginica**, the two species with genuine feature overlap — visible in each model's confusion matrix.
- Since accuracy was tied, the notebook favors the **simpler, more interpretable** model (Logistic Regression) as the declared best model, following Occam's razor — though Random Forest is noted as an equally strong, robust alternative.
- **Takeaway:** petal length and petal width alone would likely be enough to achieve high classification accuracy on this dataset; the choice between models here comes down to interpretability rather than raw performance.

## How to Run the Project

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### 2. Run the notebook
No dataset download is required — the Iris dataset ships with scikit-learn. Simply open the notebook and run all cells top to bottom:
```bash
jupyter notebook "Task_1-_Iris_Flower_Classification.ipynb"
```

The notebook will:
- Load the Iris dataset and run EDA (shape, duplicates, missing values, class balance)
- Generate pairplots, boxplots, and a correlation heatmap
- Split and scale the data
- Train and evaluate Logistic Regression, KNN, Decision Tree, and Random Forest
- Plot confusion matrices and a model accuracy comparison chart
- Print the best-performing model and its test accuracy
