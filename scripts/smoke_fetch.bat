@echo off
set PYTHONPATH=%~dp0\..\src

python --version

python - <<"PY"
import sentimental_cap_predictor as scp
print("package:", scp.__name__)
PY

python -m sentimental_cap_predictor.smoke_cli --help
python -m sentimental_cap_predictor.smoke_cli "CMD: echo hello"
