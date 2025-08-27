.PHONY: install lint format test

install:
	pip install -e .[dev]

lint:
	pre-commit run --all-files

format:
	pre-commit run ruff-format --all-files

test:
        pytest
