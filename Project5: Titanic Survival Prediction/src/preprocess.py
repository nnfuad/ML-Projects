import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/Titanic-Dataset.csv")
PROCESSED_DATA_DIR = Path("data/processed")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df



def preprocess_data(df):
    # Select relevant columns only
    df = df[
        ["Pclass", "Sex", "Age", "Fare", "Survived"]
    ]

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median()) # Fill missing ages with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    

    # Encode categorical variable
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    return df

# Split and scale
def split_and_scale(df):
    X = df.drop("Survived", axis=1) # Features
    y = df["Survived"] # Target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    ) # Stratify to maintain class distribution, random_state for reproducibility, test_size for 80-20 split, random_state for reproducibility, reproducibility='42' means 42 is used as the seed for random number generation, ensuring that the same split of data is produced each time the code is run. Could we use any other number? Yes, any integer can be used as a seed. The choice of number does not affect the quality of the split, but using the same seed allows for consistent results across runs.

    scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance. This is important for many machine learning algorithms that are sensitive to the scale of the data, such as logistic regression.
    

    X_train_scaled = scaler.fit_transform(X_train) # Transform the training data to have zero mean and unit variance. The fit_transform method computes the mean and standard deviation on the training data and then applies the transformation.
    X_test_scaled = scaler.transform(X_test) # Transform the test data using the same parameters (mean and standard deviation) computed from the training data. This ensures that the test data is scaled in the same way as the training data, which is crucial for maintaining consistency and ensuring that the model performs well on unseen data.
    # Transform is a process to center the data or to scale the data from 0 to 1.
    
    return X_train_scaled, X_test_scaled, y_train, y_test


# Save processed data
def save_processed_data(X_train, X_test, y_train, y_test):
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)



# Main execution

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale(df)
    save_processed_data(X_train, X_test, y_train, y_test)

    print("Preprocessing complete. Processed files saved.")