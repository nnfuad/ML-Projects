# Project 4: ML vs Classical Regression  
**Comparative Analysis of Machine Learning and Classical Regression Models for System Performance Prediction**

---

## 1. Overview

This project presents a systematic comparison between **classical regression techniques** and **machine learning models** for predicting **system performance metrics**, specifically program execution time, based on system configuration parameters.

The primary goal is to demonstrate:
- how model assumptions affect performance,
- when machine learning models outperform classical methods,
- and the trade-offs between interpretability and predictive accuracy.

This project is designed to be **fully reproducible**, **script-driven**, and suitable for academic evaluation.

---

## 2. Problem Statement

Modern computing systems exhibit complex, non-linear performance behavior influenced by multiple interacting parameters such as input size, number of threads, cache size, and network load.

**Objective:**  
Predict **execution time** given system configuration parameters.

---

## 3. Dataset

### 3.1 Features

| Feature Name | Description |
|-------------|-------------|
| `input_size` | Size of input data processed |
| `threads` | Number of execution threads |
| `cache_kb` | Cache size in kilobytes |
| `packet_count` | Number of network packets |

### 3.2 Target Variable

- `execution_time` (milliseconds)

### 3.3 Data Source

The dataset is **synthetically generated** to simulate realistic system behavior:
- linear growth with input size,
- diminishing returns from multithreading,
- cache saturation effects,
- stochastic noise to mimic real systems.

Synthetic data is used intentionally to:
- ensure reproducibility,
- control ground-truth relationships,
- enable fair model comparison.

Dataset generation is performed programmatically via `generate_data.py`.

---

## 4. Models Compared

### 4.1 Linear Regression
- Assumes linear, additive relationships.
- Highly interpretable.
- Limited in capturing complex interactions.

### 4.2 Polynomial Regression (Degree 2)
- Extends linear regression with interaction terms.
- Improved expressiveness.
- Prone to overfitting and poor scalability.

### 4.3 Random Forest Regression
- Non-linear, ensemble-based model.
- Captures feature interactions automatically.
- Strong generalization performance.
- Lower interpretability compared to linear models.

---

## 5. Evaluation Metrics

- **Mean Squared Error (MSE)** on a held-out test set.
- Visual comparison using **Predicted vs Actual** plots.
- Feature importance analysis (Random Forest).

---

## 6. Project Structure

```text
Project4: ML vs Classical Regression/
│
├── data/
│   ├── raw/
│   │   └── system_performance.csv
│   └── processed/
│
├── src/
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── models/
│   ├── tables/
│   │   └── mse_comparison.csv
│   └── figures/
│       ├── predicted_vs_actual_linear.png
│       ├── predicted_vs_actual_poly.png
│       └── predicted_vs_actual_rf.png
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── pipeline.sh
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 7. How to Run the Project

### 7.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 7.2 Run Full Pipeline (Recommended)

From the parent directory:

```bash
./"Project4: ML vs Classical Regression/pipeline.sh"
```

---

## 8. Results

### 8.1 Quantitative Comparison

The following table is automatically generated:

```
results/tables/mse_comparison.csv
```

Lower MSE indicates better predictive performance.

### 8.2 Visual Analysis

Each model produces a **Predicted vs Actual** plot:
- Perfect predictions lie on the diagonal.
- Scatter indicates bias and variance behavior.

Plots are saved in:
```
results/figures/
```

---

## 9. Discussion

Random Forest regression consistently outperforms linear and polynomial regression due to its ability to model non-linear relationships and feature interactions. Linear regression fails to capture diminishing returns and saturation effects, while polynomial regression improves flexibility at the cost of increased variance. These results highlight that model selection should be guided by data characteristics rather than model complexity alone.

---

## 10. Limitations

- Synthetic data does not capture all real-world system variability.
- Random Forest models extrapolate poorly beyond training ranges.
- Polynomial regression scales poorly with increasing feature count.
- Temporal and hardware-specific effects are not modeled.

---

## 11. Reproducibility

- All paths are resolved relative to script location.
- Pipeline is location-independent.
- Artifacts are generated programmatically and verified.

---

## 12. Future Work

- Cross-validation with confidence intervals
- Feature sensitivity and importance visualization
- Energy consumption or latency prediction
- Command-line argument support (`argparse`)
- Real-world benchmark datasets

---

## 13. Author

**Nur Nafis Fuad**  
Electrical & Computer Engineering Undergraduate  
Machine Learning Enthusiast

---

## 14. License

This project is intended for academic and educational use.