## Variables
ENV_PATH = .venv

## Targets
.PHONY: all build clean install execute tests

all: clean setup develop docs/build

setup:
	python3 -m venv .venv

develop: honcaml requirements.txt
	$(ENV_PATH)/bin/python -m pip install -e .

docs/build: docs/source
	$(ENV_PATH)/bin/sphinx-build -b html docs/source/ docs/build/html

install: honcaml requirements.txt
	$(ENV_PATH)/bin/python -m pip install .

tests:
	$(ENV_PATH)/bin/python -m pytest --cov=honcaml --cov-report term-missing

validate_code:
	$(ENV_PATH)/bin/flake8 --exclude=./.venv,./build

clean:
	rm -rf .venv
	rm -rf docs/build 
