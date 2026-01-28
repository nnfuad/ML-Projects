import pandas as pd
import joblib
from pathlib import Path
from utils import evaluate_model, save_predicted_vs_actual

df = pd.read_csv("../data/raw/system_performance.csv")

X = df.drop("execution_time", axis=1)
y = df["execution_time"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_dir = Path("../results/models")
fig_dir = Path("../results/figures")
table_dir = Path("../results/tables")

fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

results = []

for name in ["linear", "poly", "rf"]:
    model = joblib.load(model_dir / f"{name}.joblib")
    mse, preds = evaluate_model(model, X_test, y_test)

    save_predicted_vs_actual(
        y_test,
        preds,
        title=f"Predicted vs Actual ({name.upper()})",
        path=fig_dir / f"predicted_vs_actual_{name}.png"
    )

    results.append((name, mse))

results_df = pd.DataFrame(results, columns=["Model", "MSE"])
results_df.to_csv(table_dir / "mse_comparison.csv", index=False)

print("Evaluation complete. Results saved.")