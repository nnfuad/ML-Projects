#!/bin/bash
set -e

# Resolve project root regardless of where script is run from
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

python "$PROJECT_ROOT/src/generate_data.py"
python "$PROJECT_ROOT/src/train.py"
python "$PROJECT_ROOT/src/evaluate.py"

echo "Pipeline completed successfully."