# Loan Approval Prediction — Logistic Regression

An end-to-end machine learning pipeline that predicts whether a loan application will be approved or rejected using **Logistic Regression**.

## 📌 Problem Statement
- **Objective:** Predict loan approval status based on applicant income, assets, credit score, and other financial features
- **Dataset:** Loan Approval dataset (`loan_approval_dataset.csv`) — 4,269 records, 13 columns
- **Problem Type:** Classification
- **Target Variable:** `loan_status` (Approved / Rejected)
- **Business Relevance:** Helps financial institutions automate and speed up loan approval decisions while reducing manual review effort and human bias
- **Success Metric:** F1-score > 0.80 → **Achieved: F1 = 0.9212**

## 📊 Dataset
- **Rows:** 4,269
- **Dropped column:** `loan_id` (identifier, no predictive value)
- **Class distribution:** Approved 62.22% | Rejected 37.78% (reasonably balanced)
- **Numeric features:** `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- **Categorical features:** `education`, `self_employed`
- No missing values, no duplicate rows, no constant columns detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Joblib

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — standardized column names, dropped ID-like column (`loan_id`), checked for duplicates and constant columns
3. **Exploratory Data Analysis** — dashboard overview, class distribution check, correlation analysis, skewness check
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — checked for highly correlated features (>0.95); none found
6. **Preprocessing Pipeline** — imputation, scaling for numeric features + one-hot encoding for categorical features (via `ColumnTransformer`)
7. **Train-Test Split** — 80/20 split (3,415 train / 854 test)
8. **Model Comparison** — benchmarked 10 classification algorithms using 5-fold cross-validation
9. **Model Training** — final pipeline trained using **Logistic Regression**
10. **Model Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
11. **Sanity Checks** — train vs test score comparison to rule out overfitting
12. **Cross Validation** — 5-fold CV for robustness check
13. **Hyperparameter Tuning** — GridSearchCV & RandomizedSearchCV to tune regularization parameter `C`

## 📈 Results

| Metric | Value |
|--------|-------|
| **Accuracy (Test)** | 0.9215 |
| **Precision** | 0.9214 |
| **Recall** | 0.9215 |
| **F1-Score** | **0.9212** |
| Cross-Validation Mean Accuracy | 0.9175 (± 0.0070) |
| Best Hyperparameter (GridSearchCV) | `C = 1` |
| Train Score vs Test Score | 0.9151 vs 0.9215 (no overfitting detected) |

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Approved | 0.92 | 0.95 | 0.94 |
| Rejected | 0.92 | 0.87 | 0.89 |

**Model Comparison (Top Results — CV Accuracy):**

| Model | CV Accuracy |
|-------|-------------|
| XGBoost | 0.9841 |
| Random Forest | 0.9806 |
| Gradient Boosting | 0.9789 |
| Decision Tree | 0.9768 |
| **Logistic Regression (selected)** | 0.9175 |

## 🔑 Key Insights
- **Best Model overall:** XGBoost achieved the highest cross-validation score (98.41%), significantly outperforming Logistic Regression
- Logistic Regression was selected as the final model for its simplicity and interpretability, still achieving a strong **92.15% test accuracy**
- Train and test scores are closely aligned (0.9151 vs 0.9215), confirming the model generalizes well with no overfitting


## 📂 Folder Structure
```
Loan-Approval-Prediction/
│
├── data/
│   └── loan_approval_dataset.csv
├── logistic-regression-loan-prediction.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
