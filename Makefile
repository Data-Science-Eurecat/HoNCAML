.PHONY: build

develop:
	python -m pip install -e .

doc:
	sphinx-build -b html docs/source/ docs/build/html

install:
	python -m pip install .
