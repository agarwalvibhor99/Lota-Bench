#!/bin/bash

# Run WAH eval + save logs
# ================================

# Activate venv 
source lota-bench-venv/bin/activate

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="logs/wah"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/eval_wah_${TIMESTAMP}.log"

echo "=============================================="
echo "Running WAH Evaluation (config_wah_headless)"
echo "Timestamp : ${TIMESTAMP}"
echo "Log file  : ${LOGFILE}"
echo "=============================================="

export DISPLAY=:1

# Run evaluation and log both to terminal and file
python src/evaluate.py --config-name=config_wah_headless \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "=============================================="
echo "WAH run complete. Log saved to:"
echo "${LOGFILE}"
echo "=============================================="

