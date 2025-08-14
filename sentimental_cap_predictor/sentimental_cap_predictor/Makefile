#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sentimental_cap_predictor
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	powershell conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	powershell ruff check sentimental_cap_predictor

## Format source code with black
.PHONY: format
format:
	powershell ruff format sentimental_cap_predictor

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	powershell conda env create --name $(PROJECT_NAME) -f environment.yml
	powershell @echo ">>> conda env created. Activate with:`nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) sentimental_cap_predictor/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
