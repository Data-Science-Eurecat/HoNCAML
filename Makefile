.PHONY: build

develop:
	python -m pip install -e .

doc:
	sphinx-build -b html docs/source/ docs/build/html

install:
	python -m pip install .

tests:
	python -m pytest --cov=honcaml --cov-report term-missing honcaml/tests/data/* honcaml/tests/models/* honcaml/tests/steps/* honcaml/tests/tools/*
