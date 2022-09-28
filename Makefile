.PHONY: build

build:
	python setup.py build

doc:
	sphinx-build -b html docs/source/ docs/build/html

install:
	python setup.py install
