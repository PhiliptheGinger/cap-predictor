# Windows Task Scheduler

Automate the nightly pipeline on Windows using **Task Scheduler**.

## Environment setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install openai
   pip install -e .
   ```
2. Configure any required environment variables, for example:
   ```powershell
   $env:MLFLOW_DISABLED = "1"  # skip MLflow logging
   ```

## Batch script

Write a small batch file (e.g. `run_nightly.bat`) to activate the
environment and launch the pipeline:

```bat
@echo off
cd /d C:\path\to\cap-predictor
call venv\Scripts\activate
set MLFLOW_DISABLED=1
python -m sentimental_cap_predictor.flows.daily_pipeline run NVDA >> logs\nightly.log 2>&1
```

## Registering the task

Use `schtasks` to schedule the batch file every night at 2am:

```cmd
schtasks /Create /SC DAILY /ST 02:00 /TN "Cap Predictor Nightly" ^
  /TR "C:\path\to\cap-predictor\run_nightly.bat" /F
```

The task will run the pipeline nightly and append output to
`logs\nightly.log`. Adjust the start time, ticker symbol, or environment
variables as needed for your setup.

## Additional Resources

- [User Manual](user_manual.md) – complete usage guide
- [Documentation Index](index.md) – overview of all docs
