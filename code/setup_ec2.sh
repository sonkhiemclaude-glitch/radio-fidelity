#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results reports
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip freeze > requirements-lock.txt
.venv/bin/python code/preflight_revision.py > logs/environment.json
