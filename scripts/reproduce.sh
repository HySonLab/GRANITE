#!/usr/bin/env bash
# From repository root, with the package installed in the active environment.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
RESULTS_DIR="${1:-runs/desktop}"
python -m pytest -q
python -m granite.data --output "$RESULTS_DIR/data"
for seed in 0 1 2; do
  python -m granite.train --data "$RESULTS_DIR/data" --seed "$seed" --output "$RESULTS_DIR/seed$seed"
done
python -m granite.ablations --results "$RESULTS_DIR"
python -m granite.evaluate --results "$RESULTS_DIR"
python -m granite.verify --results "$RESULTS_DIR"
python -m granite.report --results "$RESULTS_DIR"
