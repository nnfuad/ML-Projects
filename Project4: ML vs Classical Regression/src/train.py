import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------
# Resolve project root safely
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/system_performance.csv"
MODEL_DIR = PROJECT_ROOT / "results/models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
    raise RuntimeError(
        f"Dataset missing or empty: {DATA_PATH}\n"
        "Run generate_data.py first."
    )

df = pd.read_csv(DATA_PATH)

X = df.drop("execution_time", axis=1)
y = df["execution_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Define models
# ------------------------------------------------------------------
models = {
    "linear": LinearRegression(),
    "poly": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ]),
    "rf": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

# ------------------------------------------------------------------
# Train and save
# ------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"[SAVED MODEL] {model_path.resolve()}")

print("Models trained and saved.")