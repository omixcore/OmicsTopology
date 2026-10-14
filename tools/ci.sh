#!/usr/bin/env sh
# The same checks the hosted CI runs, in the same order and with the same exit
# codes, so that a green run here means a green pipeline.  Nothing here needs
# network access or the benchmark data: the data are large, GEO-hosted and
# deliberately not committed, and every test is written against synthetic
# clouds or pure functions.
#
# Usage:
#   sh tools/ci.sh                 # uses `python` from PATH
#   PYTHON=/path/to/python sh tools/ci.sh
set -eu

PY="${PYTHON:-python}"

echo "== 1/3  install (editable, dev extra) =="
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -e ".[dev]"

echo "== 2/3  tests =="
"$PY" -m pytest -q

echo "== 3/3  generated API reference is current =="
"$PY" tools/gen_api_docs.py --check

echo "all checks passed"
