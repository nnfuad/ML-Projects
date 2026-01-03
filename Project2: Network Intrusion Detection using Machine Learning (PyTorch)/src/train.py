import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from model import IntrusionNet


# -----------------------------
# 1. Load Dataset
# -----------------------------
train_df = pd.read_csv("../data/kdd_train.csv")
test_df = pd.read_csv("../data/kdd_test.csv")

df = pd.concat([train_df, test_df])

print("Dataset Loaded")
print(df.head())


# -----------------------------
# 2. Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])


# -----------------------------
# 3. Prepare Features
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

# Convert categorical columns
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 4. Convert to PyTorch Tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# -----------------------------
# 5. Model Setup
# -----------------------------
model = IntrusionNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# 6. Training Loop
# -----------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# -----------------------------
# 7. Evaluation
# -----------------------------
with torch.no_grad():
    predictions = (model(X_test) > 0.5).float()
    accuracy = (predictions == y_test).float().mean()

print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
