import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib

df = pd.read_csv("../data/raw/system_performance.csv")

X = df.drop("execution_time", axis=1)
y = df["execution_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

model_dir = Path("../results/models")
model_dir.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, model_dir / f"{name}.joblib")

print("Models trained and saved.")