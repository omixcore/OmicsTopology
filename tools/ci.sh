#!/usr/bin/env sh
set -eu
PY="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$PY" -m pip install -q -e ".[dev]"
"$PY" -m pytest -q
"$PY" tools/gen_api_docs.py --check
"$PY" reproduce/verify.py
"$PY" reproduce/verify_v2.py --smoke
"$PY" -m omictopo --help
