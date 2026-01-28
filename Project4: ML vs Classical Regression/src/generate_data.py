import numpy as np
import pandas as pd
from pathlib import Path

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

output_path = Path("../data/raw/system_performance.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")