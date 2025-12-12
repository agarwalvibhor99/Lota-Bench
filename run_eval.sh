#!/bin/bash

# ============================================================
# Automatic Logging Runner for LLMTaskPlanning (ALFRED)
# Creates timestamped logs + prints output to terminal
# ============================================================

# Activate venv
source lota-bench-venv/bin/activate

# Timestamp for log file
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

LOGFILE="logs/eval_${TIMESTAMP}.log"

echo "=============================================="
echo "Running ALFRED Evaluation"
echo "Timestamp: ${TIMESTAMP}"
echo "Logging to: ${LOGFILE}"
echo "=============================================="

# If using DISPLAY 1
export DISPLAY=:1

# Run evaluation + save logs
python src/evaluate.py \
    --config-name=config_alfred \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "=============================================="
echo "Run Complete — Log Saved To:"
echo "${LOGFILE}"
echo "=============================================="
