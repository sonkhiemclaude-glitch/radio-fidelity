#!/usr/bin/env bash
# Rerun the original experiment suite into isolated outputs.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-../.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_NUM_THREADS
export PHYSICS_RESULTS_DIR="$PWD/../results/legacy_rerun"
export PHYSICS_FIGURES_DIR="$PWD/../reports/legacy_rerun/figures"
export PHYSICS_PAPER_DIR="$PWD/../reports/legacy_rerun/tables"
mkdir -p "$PHYSICS_RESULTS_DIR" "$PHYSICS_FIGURES_DIR" "$PHYSICS_PAPER_DIR"

echo "=== 0. data preparation ==="
[ -f ../data/lorawan.npz ]  || $PY -u prep_lorawan.py
[ -f ../data/geometry.npz ] || $PY -u prep_osm.py
[ -f ../data/uji.npz ]      || $PY -u prep_uji.py

echo "=== 1. theory self-checks ==="
$PY -u test_theory.py

echo "=== 2. censored maximum-likelihood ladder fit ==="
[ -f ../data/tobit_fits.npz ] || $PY -u fit_ladder.py

echo "=== 3. experiments ==="
for e in exp1_premise exp2_ladder exp3_criterion exp4_admission \
         exp5_unlabelled exp6_foundation exp7_indoor exp8_robustness \
         exp9_stability exp10_mechanism; do
  echo "--- $e ---"
  $PY -u $e.py
done

echo "=== 4. figures and numbers ==="
$PY -u figures.py
$PY -u tables.py
$PY -u mknumbers.py

echo "Legacy rerun outputs are in results/legacy_rerun and reports/legacy_rerun."
echo "Use code/run_revision.sh from the project root to run the additional experiments."
