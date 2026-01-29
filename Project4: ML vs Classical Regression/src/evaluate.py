import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# Resolve project root safely (NO reliance on working directory)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/raw/system_performance.csv"
MODEL_DIR = PROJECT_ROOT / "results/models"
FIG_DIR = PROJECT_ROOT / "results/figures"
TABLE_DIR = PROJECT_ROOT / "results/tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("execution_time", axis=1)
y = df["execution_time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------
def save_plot(y_true, y_pred, title, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------------------------------------------------
# Evaluate models
# ------------------------------------------------------------------
results = []

for name in ["linear", "poly", "rf"]:
    model_path = MODEL_DIR / f"{name}.joblib"
    model = joblib.load(model_path)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    results.append((name, mse))

    fig_path = FIG_DIR / f"predicted_vs_actual_{name}.png"
    save_plot(
        y_test,
        preds,
        f"Predicted vs Actual ({name.upper()})",
        fig_path
    )

    print(f"[SAVED] {fig_path.resolve()}  | size = {fig_path.stat().st_size} bytes")

# ------------------------------------------------------------------
# Save and reload table
# ------------------------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "MSE"])
table_path = TABLE_DIR / "mse_comparison.csv"
results_df.to_csv(table_path, index=False)

print(f"[SAVED] {table_path.resolve()}  | size = {table_path.stat().st_size} bytes")

# Reload to verify
reloaded_df = pd.read_csv(table_path)
print("\nReloaded table:")
print(reloaded_df)