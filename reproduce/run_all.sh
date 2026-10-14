#!/usr/bin/env sh
# Third-party reproduction in one command, from a fresh clone:
#
#   sh reproduce/run_all.sh
#
# What it does, in order:
#   1. installs the package (editable, with dev extras -- pytest is the
#      executable specification),
#   2. optionally installs the one optional baseline dependency (umap-learn);
#      pass BASELINES=1 to include it, the default run does not need it,
#   3. runs the test suite,
#   4. runs the known-topology demo (every case's answer is written down
#      before the code runs),
#   5. recomputes every published headline number from the committed raw
#      benchmark records and compares them with what the paper says
#      (reproduce/verify.py).
#
# Nothing here touches the network: the GEO-hosted data are deliberately not
# committed and every published number is either recomputed from committed
# raw records or measured on synthetic clouds generated with pinned seeds.
set -eu

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== 1/4  install =="
"$PY" -m pip install -q -e ".[dev]"
if [ "${BASELINES:-0}" = "1" ]; then
    "$PY" -m pip install -q -e ".[baselines]"
fi

echo "== 2/4  test suite (the executable specification) =="
"$PY" -m pytest -q

echo "== 3/4  known-topology demo =="
"$PY" examples/demo_known_shapes.py

echo "== 4/4  independent recomputation of the published numbers =="
"$PY" reproduce/verify.py

echo
echo "reproduction complete: every check passed"
