# Email Spam Detection using Machine Learning

## Project Overview
This project focuses on building a **machine learning–based Email Spam Detection System** to classify email messages as **spam or legitimate (ham)**. The objective is to demonstrate how classical machine learning techniques can be effectively applied to **text-based cybersecurity problems**.

This project combines:
- Natural Language Processing (NLP)
- Supervised machine learning
- Text feature engineering (TF-IDF)
- Model evaluation and visualization
- Clean and reproducible ML experimentation

---

## Problem Statement
Email spam poses a persistent security and productivity challenge. Traditional rule-based filters struggle to generalize against **new spam patterns and adversarial wording**. This project applies **supervised machine learning** to automatically learn discriminative patterns from email content.

**Task Type:** Binary Classification  
**Classes:**  
- Ham (0) — Legitimate Email  
- Spam (1) — Unwanted / Malicious Email  

---

## Dataset
- **Dataset:** Spam Email Dataset  
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/mfaisalqureshi/spam-email
- **Domain:** Email Security / NLP
- **Features:** Email message content (text)
- **Target:** Email category (spam / ham)

> Note: This dataset includes **email bodies only** and does not contain header metadata such as sender, subject, or URLs.

---

## Tools & Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- TF-IDF (Text Feature Extraction)

---

## Methodology

### 1. Data Loading
```python
import pandas as pd

df = pd.read_csv("spam.csv")
print(df.head())
```

---

### 2. Data Exploration
```python
print(df.columns)
print(df['Category'].value_counts())
print(df.info())
```

---

### 3. Data Preprocessing
```python
from sklearn.model_selection import train_test_split

df = df.rename(columns={"Category": "label", "Message": "text"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### 4. Text Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

---

### 5. Model Selection
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

---

### 6. Training & Evaluation
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test_vec)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

### 7. Visualization
```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0,1], ["Ham", "Spam"])
plt.yticks([0,1], ["Ham", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()
```

---

## Results
- Achieved approximately **97% accuracy** on the test set
- Strong spam detection performance
- Visual analysis highlights misclassification patterns

---

## Limitations
- No semantic understanding of language
- No email metadata used
- Vulnerable to adversarial wording

---

## Future Improvements
- Logistic Regression / SVM comparison
- ROC and Precision–Recall curves
- Keyword importance analysis
- Transformer-based models (BERT)

---

## How to Run
```bash
cd app
python train.py
```

---

## Repository Structure
```
├── data/
│   └── spam.csv
├── notebooks/
│   └── intrusion_detection.ipynb
├── src/
│   ├── model.py
│   └── train.py
├── README.md
```

---

## Author
**Name:** Nur Nafis Fuad  
**Field:** Electrical & Computer Engineering | ML & Cybersecurity  
**Country:** Bangladesh
