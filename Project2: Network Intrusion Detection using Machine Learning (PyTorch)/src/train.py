# train.py
# Network Intrusion Detection using PyTorch
# Fixes: binary labels, categorical encoding, tensor conversion

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import IntrusionNet  # Make sure model.py is in the same folder


# -----------------------------
# 1. Load Dataset
# -----------------------------
train_df = pd.read_csv("../data/kdd_train.csv")
test_df = pd.read_csv("../data/kdd_test.csv")

df = pd.concat([train_df, test_df], ignore_index=True)

print("Dataset Loaded")
print(df.head())


# -----------------------------
# 2. Convert Labels to Binary
# -----------------------------
# NSL-KDD has multiple attack types. For binary classification:
# normal -> 0, all attacks -> 1
df["label"] = df["labels"].apply(lambda x: 0 if x == "normal" else 1)

print("Labels Converted to Binary")
print(df["label"].value_counts())


# -----------------------------
# 3. Prepare Features
# -----------------------------
X = df.drop(['labels', 'label'], axis=1)
y = df['label']

# Convert categorical features to one-hot
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 4. Convert to PyTorch Tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# -----------------------------
# 5. Model Setup
# -----------------------------
model = IntrusionNet(X_train_tensor.shape[1])  # input_dim = number of features
criterion = nn.BCELoss()                        # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# 6. Training Loop
# -----------------------------
epochs = 20
loss_values = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# -----------------------------
# 7. Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = (model(X_test_tensor) > 0.5).float()
    accuracy = (predictions == y_test_tensor).float().mean()

print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")


# -----------------------------
# 8. Optional: Loss Curve Visualization
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()