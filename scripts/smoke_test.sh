#!/usr/bin/env bash
# Lightweight smoke test: ensure Python runs, package imports, config loads,
# CLI help works and a simple module executes without heavy dependencies.
set -euo pipefail

export PYTHONPATH="$(dirname "$0")/../src"

python3 --version

python3 - <<'PY'
import sentimental_cap_predictor as scp
print("package:", scp.__name__)
PY

python3 -m sentimental_cap_predictor.smoke_cli --help
python3 -m sentimental_cap_predictor.smoke_cli "CMD: echo hello"
