## Variables
SHELL = bash
ENV_PATH = .venv

## Targets
.PHONY: all build clean install tests

all: clean setup develop docs/build

setup:
	python3 -m venv .venv

build: dist

dist: honcaml
	python3 -m build

develop: honcaml pyproject.toml
	$(ENV_PATH)/bin/python -m pip install -e .
	$(ENV_PATH)/bin/python -m pip install honcaml[check] honcaml[document] honcaml[tests]

docs/build: docs/source
	$(ENV_PATH)/bin/sphinx-build -b html docs/source/ docs/build/html

install: honcaml pyproject.toml
	$(ENV_PATH)/bin/python -m pip install .

tests:
	$(ENV_PATH)/bin/python -m pytest --cov=honcaml --cov-report term-missing

validate_code:
	$(ENV_PATH)/bin/flake8 --exclude=./.venv,./build

clean:
	rm -rf .venv
	rm -rf docs/build
