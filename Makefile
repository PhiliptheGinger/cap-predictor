.PHONY: install lint format test streamlit

install:
	pip install -e .[dev]

lint:
	pre-commit run --all-files

format:
	pre-commit run ruff-format --all-files

test:
        pytest

streamlit:
        streamlit run src/sentimental_cap_predictor/ui/streamlit_app.py
