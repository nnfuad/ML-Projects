import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from model import build_model


# -----------------------------
# 1. Load and prepare dataset
# -----------------------------
df = pd.read_csv("../data/spam.csv")

# Rename columns to standard names
df = df.rename(columns={
    "Category": "label",
    "Message": "text"
})

# Encode labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

if df["label"].isnull().any():
    raise ValueError("Label encoding failed. Check dataset labels.")

# -----------------------------
# 2. Visualize class distribution
# -----------------------------
label_counts = df["label"].value_counts()

plt.figure()
plt.bar(["Ham", "Spam"], label_counts.values)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Class")
plt.show()

# -----------------------------
# 3. Train-test split
# -----------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Build and train model
# -----------------------------
model, vectorizer = build_model()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model.fit(X_train_vec, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# -----------------------------
# 6. Confusion Matrix Visualization
# -----------------------------
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()

# -----------------------------
# 7. Save trained model
# -----------------------------
with open("spam_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)