# 📧 Email Spam Detection with Machine Learning

## Objective
Build a Natural Language Processing (NLP) binary classifier that distinguishes **spam** messages from **legitimate (ham)** messages using classic machine learning models, and compare their performance on an imbalanced real-world dataset.

## Technologies Used
- **Python** — core language
- **pandas / numpy** — data loading and manipulation
- **NLTK / re** — text preprocessing (stopword removal, stemming, tokenization)
- **scikit-learn** — TF-IDF vectorization, Multinomial Naive Bayes, Logistic Regression, Linear SVM, evaluation metrics
- **matplotlib / seaborn / wordcloud** — visualization and EDA

## Dataset Information
- **Source file:** `spam.csv` (SMS Spam Collection dataset)
- **Size:** 5,572 messages (5,169 after removing 403 duplicate rows)
- **Labels:** `ham` (legitimate) vs `spam` (unsolicited/promotional)
- **Class balance:** ~87% ham, ~13% spam — an **imbalanced** dataset
- **Raw columns:** `v1` (label), `v2` (message text), plus three empty artifact columns (`Unnamed: 2/3/4`) dropped during cleaning
- Columns were renamed to `target` and `message` for clarity

## Steps / Workflow
1. **Load & inspect the data** — read the CSV, drop empty artifact columns, rename columns
2. **Class distribution analysis** — confirm the ~87/13 ham/spam imbalance
3. **Duplicate removal** — dropped 403 duplicate rows (5,572 → 5,169 rows)
4. **Text preprocessing pipeline** (`preprocess_text`):
   - Lowercase conversion
   - Punctuation & digit removal
   - Tokenization (whitespace split)
   - Stopword removal (NLTK English stopwords)
   - Stemming (Porter Stemmer)
5. **Feature engineering for EDA** — character count, word count, sentence count per message
6. **Exploratory Data Analysis (EDA)**:
   - Distribution plots of message length (characters/words/sentences) by class
   - Boxplots and correlation heatmap of length features
   - WordClouds comparing frequent spam vs. ham vocabulary
7. **Feature extraction** — TF-IDF vectorization (`max_features=3000`) on cleaned text
8. **Train/test split** — 80/20 split, stratified on the target to preserve class ratio
9. **Model training** — trained and evaluated three classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
10. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrices, and full classification reports for each model
11. **Discussion** — analysis of why recall is critical for spam detection, with attention to the precision/recall trade-off

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Linear SVM** | **0.9778** | 0.9737 | **0.8473** | **0.9061** |
| Naive Bayes | 0.9729 | **0.9905** | 0.7939 | 0.8814 |
| Logistic Regression | 0.9555 | **1.0000** | 0.6489 | 0.7870 |

*(Test set: 1,034 messages — 903 ham, 131 spam)*

## Results
- All three models achieved **>95% accuracy**, but accuracy alone is misleading on this imbalanced dataset (87% ham baseline).
- **Naive Bayes and Logistic Regression** achieved very high precision (few false positives — legitimate mail rarely misclassified as spam), but at the cost of lower recall (more spam slipping through), especially Logistic Regression (65% recall).
- **Linear SVM performed best overall**, achieving the highest recall (84.7%) and the best F1-score (0.906) — the strongest balance between catching spam and preserving legitimate mail.
- **Key takeaway:** for a spam filter, high precision (rarely flagging real mail as spam) is essential, but recall must also be kept as high as possible without sacrificing that precision — which is why Linear SVM is recommended as the final model.

## How to Run the Project

### 1. Clone/download the project and install dependencies
```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

### 2. Download required NLTK data
The notebook downloads these automatically on first run, but you can also fetch them manually:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

### 3. Set the dataset path
Update the `DATA_PATH` variable in the notebook to point to your local copy of `spam.csv`:
```python
DATA_PATH = "path/to/spam.csv"
```

### 4. Run the notebook
Open `Task_4_-_Email_Spam_Detection_with_Machine_Learning.ipynb` in Jupyter Notebook / JupyterLab / VS Code and run all cells top to bottom:
```bash
jupyter notebook Task_4_-_Email_Spam_Detection_with_Machine_Learning.ipynb
```

The notebook will:
- Clean and preprocess the text
- Generate EDA plots and wordclouds
- Vectorize text with TF-IDF
- Train and evaluate Naive Bayes, Logistic Regression, and Linear SVM
- Output comparison tables, confusion matrices, and classification reports
