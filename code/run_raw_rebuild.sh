#!/usr/bin/env bash
# Dated-map reconstruction, isolated from the submission caches and main tables.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
mkdir -p data/from_raw results/raw_rebuild reports/raw_rebuild
export REV_DATA_DIR="$PWD/data/from_raw"
export PHYSICS_DATA_DIR="$PWD/data/from_raw"
export PHYSICS_RESULTS_DIR="$PWD/results/raw_rebuild"
export PHYSICS_PAPER_DIR="$PWD/reports/raw_rebuild"
$PY code/prep_lorawan.py
$PY code/prep_uji.py
$PY code/prep_osm.py
$PY code/fit_ladder.py
$PY code/exp1_premise.py
$PY code/tables.py
