#!/bin/bash

source lota-bench-venv/bin/activate

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p logs
LOGFILE="logs/eval_${TIMESTAMP}.log"

echo "Running ALFRED Evaluation, logging to: ${LOGFILE}"

xvfb-run -s "-screen 0 1400x900x24" \
  python src/evaluate.py \
    --config-name=config_alfred \
    2>&1 | tee "${LOGFILE}"

echo "Done. Log: ${LOGFILE}"
