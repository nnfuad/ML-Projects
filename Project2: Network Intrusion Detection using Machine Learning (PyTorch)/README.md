# Network Intrusion Detection using Machine Learning (PyTorch)

## Project Overview
This project focuses on building a **machine learning–based Network Intrusion Detection System (NIDS)** to classify network traffic as **normal or malicious**. The goal is to demonstrate how ML can be applied to cybersecurity problems using structured network data.

This project combines:
- Cybersecurity fundamentals
- Machine learning on tabular data
- PyTorch-based model development
- Proper experimental documentation

---

## Problem Statement
Traditional rule-based intrusion detection systems struggle to detect **new or evolving attacks**. This project applies **supervised machine learning** to identify malicious traffic patterns based on statistical network features.

**Task Type:** Binary Classification  
**Classes:** Normal Traffic (0), Attack Traffic (1)

---

## Dataset
- **Dataset:** NSL-KDD (Improved version of KDD Cup 99)
- **Domain:** Network Security
- **Features:** Protocol type, service, flag, packet statistics, error rates, etc.
- **Target:** Traffic label (normal / attack)

> The NSL-KDD dataset is widely used in academic research and avoids redundancy issues present in the original KDD99 dataset.

---

## Tools & Technologies
- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

### 1. Data Loading
```python
import pandas as pd

df = pd.read_csv('nsl_kdd.csv')
print(df.head())
```

---

### 2. Data Exploration
- Check class imbalance
- Inspect feature distributions
- Identify categorical vs numerical features

```python
print(df['label'].value_counts())
print(df.info())
```

---

### 3. Data Preprocessing
- Encode categorical features
- Normalize numerical features
- Convert labels to binary format

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 4. Convert to PyTorch Tensors
```python
import torch

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)
```

---

### 5. Model Architecture
A fully connected neural network is used for classification.

```python
import torch.nn as nn

class IntrusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = IntrusionNet(X_train.shape[1])
```

---

### 6. Training Setup
```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

### 7. Model Training
```python
loss_values = []

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

### 8. Evaluation
```python
from sklearn.metrics import accuracy_score, confusion_matrix

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, y_pred_class)

print(f"Test Accuracy: {accuracy*100:.2f}%")
```

---

### 9. Visualization
```python
import matplotlib.pyplot as plt

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
```

---

## Results
- Achieved strong classification accuracy on test data
- Model effectively learns attack vs normal traffic patterns
- Demonstrates the feasibility of ML-based intrusion detection

---

## Why This Project is CV / Scholarship Worthy
- Aligns **Machine Learning + Cybersecurity**
- Uses a **standard research dataset** (NSL-KDD)
- Demonstrates full ML pipeline
- Easy to extend into research or thesis work

---

## Future Improvements
- Multi-class attack classification
- Use LSTM for sequential traffic modeling
- Compare with classical ML models (SVM, Random Forest)
- Add ROC-AUC and Precision-Recall analysis

---

## How to Run
1. Clone the repository
2. Install dependencies
3. Run the notebook or Python script

---

## Repository Structure
```
├── data/
│   └── nsl_kdd.csv
├── notebooks/
│   └── intrusion_detection.ipynb
├── src/
│   └── model.py
├── README.md
```

---

## Author
**Name:** Nur Nafis Fuad  
**Field:** Electrical & Computer Engineering | ML & Cybersecurity  
**Country:** Bangladesh

---
