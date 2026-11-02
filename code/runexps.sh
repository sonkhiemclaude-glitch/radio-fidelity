#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-../.venv/bin/python}
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
for e in "$@"; do
  echo "########## $e ##########"
  nice -n 5 $PY -u $e.py || echo "!!!!! $e FAILED"
done
echo "########## ALL DONE ##########"
