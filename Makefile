## Variables
SHELL = bash
ENV_PATH = .venv

## Targets
.PHONY: all build clean install uninstall tests validate_code

all: clean setup develop

setup:
	python3 -m venv .venv

upload: dist
	python3 -m twine upload dist/*

dist: honcaml pyproject.toml
	rm -rf dist/*
	python3 -m build

develop: honcaml pyproject.toml uninstall
	$(ENV_PATH)/bin/python -m pip install -e .
	$(ENV_PATH)/bin/python -m pip install honcaml[check] honcaml[document] honcaml[tests]

docs/build: docs/source
	sphinx-build -b html docs/source/ docs/build/html

install: honcaml pyproject.toml uninstall
	$(ENV_PATH)/bin/python -m pip install .

uninstall:
	$(ENV_PATH)/bin/python -m pip uninstall honcaml

tests:
	$(ENV_PATH)/bin/python -m pytest --cov=honcaml --cov-report term-missing

validate_code:
	$(ENV_PATH)/bin/flake8 --exclude=./.venv,./build

clean:
	rm -rf .venv
	rm -rf docs/build
