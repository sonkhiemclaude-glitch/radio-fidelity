#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
mkdir -p results logs data/from_raw
$PY code/test_theory.py
$PY code/test_revision.py
$PY code/test_vector_geometry.py
$PY code/test_revision_math.py
$PY code/test_spatial_bootstrap.py
$PY code/download_data.py
REV_DATA_DIR="$PWD/data/from_raw" $PY code/prep_lorawan.py
REV_DATA_DIR="$PWD/data/from_raw" $PY code/prep_uji.py
$PY code/validate_raw.py
bash code/run_raw_rebuild.sh > logs/raw_rebuild.log 2>&1
$PY -u code/exp11_cluster_admission.py > logs/exp11.log 2>&1 &
CPU_ADMISSION=$!
$PY -u code/exp12_geometry.py > logs/exp12.log 2>&1 &
CPU_GEOMETRY=$!
$PY -u code/exp13_models.py > logs/exp13.log 2>&1
wait "$CPU_ADMISSION"
wait "$CPU_GEOMETRY"
$PY -u code/exp16_vector_geometry.py > logs/exp16.log 2>&1 &
CPU_VECTOR=$!
$PY -u code/exp14_downstream_admission.py > logs/exp14.log 2>&1
$PY -u code/exp15_indoor_holdout.py > logs/exp15.log 2>&1
wait "$CPU_VECTOR"
$PY -u code/exp17_map_localizer.py > logs/exp17.log 2>&1
$PY -u code/exp18_spatial_bootstrap.py > logs/exp18.log 2>&1
$PY code/validate_revision.py
$PY code/export_results.py
mkdir -p reports/tables_latex
PHYSICS_PAPER_DIR="$PWD/reports/tables_latex" $PY code/tables.py
