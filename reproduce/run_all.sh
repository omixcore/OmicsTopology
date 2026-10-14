#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$PY" -m pip install -q -e ".[dev]"
"$PY" reproduce/reproduce_v2.py "$@"
