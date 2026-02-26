#!/bin/bash

# ----------------------------------
# ML Pipeline: Titanic Survival
# ----------------------------------

set -e  # exit immediately if a command fails

echo "======================================"
echo "Starting ML pipeline..."
echo "======================================"

# Step 1: Preprocessing
echo "[1/3] Running preprocessing..."
python src/preprocess.py
echo "Preprocessing completed."
echo "--------------------------------------"

# Step 2: Training
echo "[2/3] Training model..."
python src/train.py
echo "Model training completed."
echo "--------------------------------------"

# Step 3: Evaluation
echo "[3/3] Evaluating model..."
python src/evaluate.py
echo "Evaluation completed."
echo "--------------------------------------"

echo "======================================"
echo "Pipeline finished successfully."
echo "Results saved in results/metrics.txt"
echo "======================================"