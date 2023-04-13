## Variables

ENV_BIN = .venv/bin
EX_PIPELINE_DIR = config/pipelines/examples
TEST_DIR = honcaml/tests
PROBLEMS = classification regression
STEPS = benchmark train predict

# Automatically generated variables for specific debug and test targets
# For example, it is possible to run: make debug_classification_train,
# or make test_regression_benchmark

EXAMPLE_TARGETS := $(foreach step, $(STEPS),\
	$(foreach problem, $(PROBLEMS), $(problem)_$(step)))
DEBUG_EXAMPLE_TARGETS := $(foreach target, $(EXAMPLE_TARGETS), debug_$(target))
TEST_EXAMPLE_TARGETS := $(foreach target, $(EXAMPLE_TARGETS), test_$(target))

## Targets

.PHONY: build doc install debug_$(EXAMPLE_TARGETS) \
		test_$(EXAMPLE_TARGETS) run_tests

develop: honcaml config/requirements.txt
	$(ENV_BIN)/python -m pip install -e .

doc: docs/source
	sphinx-build -b html docs/source/ docs/build/html

install: honcaml config/requirements.txt
	$(ENV_BIN)/python -m pip install .

$(DEBUG_EXAMPLE_TARGETS):
	$(ENV_BIN)/python -m pdb -m honcaml.__main__ -c $(EX_PIPELINE_DIR)/$(subst debug_,,$(@)).yaml

$(TEST_EXAMPLE_TARGETS):
	$(ENV_BIN)/honcaml -c $(EX_PIPELINE_DIR)/$(subst test_,,$(@)).yaml

run_tests:
	$(ENV_BIN)/python -m pytest --cov=honcaml --cov-report term-missing
