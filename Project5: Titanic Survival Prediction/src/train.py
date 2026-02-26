import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    return X_train, y_train



def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000) # max_iter is set to 1000 to ensure convergence, especially if the dataset is large or has many features. Logistic regression can sometimes require more iterations to converge, and setting a higher max_iter helps prevent convergence warnings.
    model.fit(X_train, y_train) # Fit the logistic regression model to the training data.
    return model


# Main execution
if __name__ == "__main__":
    X_train, y_train = load_processed_data()
    model = train_model(X_train, y_train)

    dump(model, MODEL_PATH)

    print(f"Model trained and saved at {MODEL_PATH}")