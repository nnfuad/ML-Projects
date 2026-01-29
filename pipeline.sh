#!/bin/bash
set -e

python src/generate_data.py
python src/train.py
python src/evaluate.py

echo "Pipeline completed successfully."