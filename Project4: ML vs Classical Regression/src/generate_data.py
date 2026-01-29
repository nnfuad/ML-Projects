from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Resolve project root safely
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/system_performance.csv"

np.random.seed(42)
N = 1000

input_size = np.random.uniform(100, 10000, N)
threads = np.random.randint(1, 16, N)
cache_kb = np.random.uniform(128, 8192, N)
packet_count = np.random.uniform(0, 1e5, N)

execution_time = (
    0.002 * input_size
    - 8 * np.log(threads)
    - 0.01 * np.sqrt(cache_kb)
    + 0.00005 * packet_count
    + np.random.normal(0, 20, N)
)

df = pd.DataFrame({
    "input_size": input_size,
    "threads": threads,
    "cache_kb": cache_kb,
    "packet_count": packet_count,
    "execution_time": execution_time
})

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_PATH, index=False)

print(f"Dataset saved to {DATA_PATH.resolve()}")