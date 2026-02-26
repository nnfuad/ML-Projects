# Project 5: Titanic Survival Prediction
**End-to-End Binary Classification Pipeline for Survival Prediction Using Classical Machine Learning**

---

## 1. Overview

This project implements a **complete, reproducible binary classification pipeline** to predict passenger survival on the Titanic using classical machine learning techniques.

The primary objective of this project is to demonstrate:
- correct machine learning workflow design,
- disciplined separation of data processing, training, and evaluation,
- appropriate metric selection for imbalanced classification,
- and clean, script-driven experimentation suitable for production-oriented ML roles.

The project intentionally emphasizes **clarity, correctness, and interpretability** over model complexity.

---

## 2. Problem Statement

The sinking of the RMS Titanic is a well-known historical event in which passenger survival depended on multiple demographic and socio-economic factors.

**Objective:**  
Given passenger attributes such as class, sex, age, and fare, predict whether a passenger survived (`1`) or did not survive (`0`).

This task is framed as a **binary classification problem**.

---

## 3. Dataset

### 3.1 Data Source

The dataset used is the publicly available **Titanic Dataset**, provided as:
The dataset contains passenger-level records with demographic and ticket information.

---

### 3.2 Selected Features

A reduced and interpretable feature set is used for the baseline model:

| Feature Name | Description |
|-------------|-------------|
| `Pclass` | Passenger ticket class (1st, 2nd, 3rd) |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `Fare` | Ticket fare paid |

---

### 3.3 Target Variable

- `Survived`
  - `1` → Passenger survived
  - `0` → Passenger did not survive

---

### 3.4 Data Characteristics

- Mild class imbalance between survivors and non-survivors
- Missing values present in the `Age` feature
- Mixture of numerical and categorical variables

These characteristics motivate careful preprocessing and evaluation metric selection.

---

## 4. Methodology

### 4.1 Exploratory Data Analysis (EDA)

EDA is performed separately in a dedicated Jupyter notebook and includes:
- target variable distribution analysis,
- missing value inspection,
- numerical feature distribution analysis,
- categorical feature relationship with survival.

EDA is intentionally restricted to **data understanding only**.  
No preprocessing or model training is performed in the notebook to prevent data leakage.

---

### 4.2 Preprocessing Pipeline

Preprocessing is implemented in `preprocess.py` and includes:

- column selection,
- median imputation for missing `Age` values,
- binary encoding of the `Sex` feature,
- train–test split with stratification,
- feature scaling using `StandardScaler`.

All preprocessing steps are executed **after splitting** to prevent information leakage.

---

### 4.3 Model Selection

#### Logistic Regression (Baseline Model)

Logistic Regression is selected as the baseline model because:
- it is well-suited for binary classification,
- it provides interpretable coefficients,
- it serves as a strong benchmark before introducing more complex models.

The model is trained using default hyperparameters with an increased iteration limit to ensure convergence.

---

## 5. Evaluation Metrics

Due to class imbalance, model performance is evaluated using:

- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

Accuracy is intentionally not used as the primary metric, as it can be misleading in imbalanced classification settings.

Evaluation is performed on a held-out test set and saved as a persistent artifact.

---

## 6. Project Structure

```text
Project5: Titanic Survival Prediction/
│
├── data/
│   ├── raw/
│   │   └── Titanic-Dataset.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── logistic_regression.joblib
│
├── results/
│   └── metrics.txt
│
├── pipeline.sh
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. How to Run the Project

### 7.1 Install Dependencies

From the project directory, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 7.2 Run Full Pipeline (Recommended)

Execute the complete machine learning pipeline in the correct order:

```bash
./pipeline.sh
```

This pipeline performs:
1. Data preprocessing  
2. Model training  
3. Model evaluation  

All outputs are generated automatically and saved as artifacts.

---

## 8. Results

### 8.1 Quantitative Results

Model evaluation metrics are saved to:

```
results/metrics.txt
```

The reported metrics include:
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

All metrics are computed on a held-out test set.

### 8.2 Interpretation

The Logistic Regression baseline model demonstrates reasonable predictive performance given the limited and interpretable feature set. The results align with known survival patterns, particularly the strong influence of passenger gender and ticket class on survival probability.

---

## 9. Discussion

This project demonstrates that correct preprocessing and evaluation are often more important than model complexity. A simple baseline model, when applied with disciplined data handling and appropriate metric selection, can provide reliable and interpretable results. The separation of concerns between preprocessing, training, and evaluation helps avoid common machine learning pitfalls such as data leakage and metric misuse.

---

## 10. Limitations

- The feature set is intentionally minimal, which limits maximum achievable performance  
- No hyperparameter tuning was performed  
- Cross-validation was not included  
- The dataset reflects historical and socio-economic biases present in the original data  

---

## 11. Reproducibility

- All scripts use relative paths  
- No hidden state or notebook-dependent logic  
- All artifacts (processed data, models, metrics) are generated programmatically  
- Pipeline execution is deterministic given fixed random seeds  

---

## 12. Future Work

- Cross-validation with confidence intervals  
- Tree-based and ensemble models (Random Forest, Gradient Boosting)  
- Feature importance and sensitivity analysis  
- Fairness and bias evaluation  
- Command-line argument support using `argparse`  

---

## 13. Author

**Nur Nafis Fuad**  
Electrical & Computer Engineering Undergraduate  
Machine Learning Enthusiast  

---

## 14. License

This project is intended for academic, educational, and demonstration purposes.
