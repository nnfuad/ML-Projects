import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from joblib import load
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/logistic_regression.joblib")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "metrics.txt"


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    return X_test, y_test


def load_model():
    return load(MODEL_PATH)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test) # model.predict is used to generate predicted class labels for the test data (X_test) based on the trained logistic regression model. The output y_pred will contain the predicted class labels (0 or 1) for each instance in the test set.
    y_proba = model.predict_proba(X_test)[:, 1] # model.predict_proba is used to generate predicted probabilities for each class (0 and 1) for the test data (X_test). The output is a 2D array where each row corresponds to an instance in the test set, and the columns represent the probabilities of belonging to class 0 and class 1.
    #By using [:, 1], we are selecting the probabilities for class 1 (survived) for each instance, which is what we need to calculate the ROC-AUC score. The resulting y_proba will be a 1D array containing the predicted probabilities of survival for each instance in the test set.

    report = classification_report(y_test, y_pred) # classification_report is a function from sklearn.metrics that generates a text report showing the main classification metrics (precision, recall, f1-score, and support) for each class in the target variable (y_test) based on the predicted class labels (y_pred). The output report will provide insights into the performance of the model for both classes (survived and not survived) in terms of how well it is predicting each class.
    roc_auc = roc_auc_score(y_test, y_proba) # roc_auc_score is a function from sklearn.metrics that calculates the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score for binary classification problems. It takes the true binary labels (y_test) and the predicted probabilities for the positive class (y_proba) as input and returns a single scalar value representing the ROC-AUC score. The ROC-AUC score ranges from 0 to 1, where a score of 1 indicates perfect classification, a score of 0.5 indicates random guessing, and a score below 0.5 indicates worse than random performance. In this context, it will help us evaluate how well our logistic regression model is distinguishing between survivors and non-survivors based on the predicted probabilities.

    return report, roc_auc



def save_results(report, roc_auc):
    with open(RESULTS_PATH, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    X_test, y_test = load_test_data()
    model = load_model()

    report, roc_auc = evaluate_model(model, X_test, y_test)
    save_results(report, roc_auc)

    print("Evaluation complete. Metrics saved.")