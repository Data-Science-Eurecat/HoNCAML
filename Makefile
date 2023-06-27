## Variables
ENV_BIN = .venv/bin

## Targets
.PHONY: build install execute $(DEBUG_EXAMPLE_TARGETS) $(TEST_EXAMPLE_TARGETS) run_tests

develop: honcaml requirements.txt
	$(ENV_BIN)/python -m pip install -e .

docs/build: docs/source
	$(ENV_BIN)/sphinx-build -b html docs/source/ docs/build/html

install: honcaml config/requirements.txt
	$(ENV_BIN)/python -m pip install .

run_tests:
	$(ENV_BIN)/python -m pytest --cov=honcaml --cov-report term-missing

validate_code:
	$(ENV_BIN)/flake8 --exclude=./.venv,./build
