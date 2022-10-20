# HoNCAML

Description
-----------

Holistic and No Code Auto Machine Learning

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE.md
    ├── README.md
    ├── bin
    ├── config
    ├── data    
    │   ├── external
    │   ├── interim
    │   ├── logs
    │   ├── models
    │   ├── processed
    │   └── raw    
    ├── docs
    ├── notebooks
    ├── reports
    └── src
        ├── data
        ├── models
        ├── tools
        ├── visualization
        └── tests

## Generate documentation

From project root and inside the virtual environment, execute the following
commands*:

1. Install python package

   ```commandline
   make install
   ```

2. Generate static documentation

    ```commandline
    make doc
    ```

*For details, see `Makefile` file in project root directory.

Afterwards, opening in any browser the local file:
`file:///{project-dir}/honcaml/docs/build/html/index.html`,
replacing `project-dir` by current project directory, should be enough to see
the documentation.

## Tests
Comanda simple (correr un fitxer de tests):
```
python -m pytest honcaml/tests/data/transform.py
```

Comanda per correr una carpeta de tests:
```
python -m pytest honcaml/tests/data/*
```

Comanda per correr totes les carpetes de tests:
```
python -m pytest honcaml/tests/data/* honcaml/tests/models/* honcaml/tests/steps/* honcaml/tests/tools/*
```

Comanda per correr totes les carpetes de tests + coverage:
```
python -m pytest --cov=honcaml honcaml/tests/data/* honcaml/tests/models/* honcaml/tests/steps/* honcaml/tests/tools/*
```

Comanda per correr totes les carpetes de tests + coverage + missing lines:
```
python -m pytest --cov=honcaml --cov-report term-missing honcaml/tests/data/* honcaml/tests/models/* honcaml/tests/steps/* honcaml/tests/tools/*
```
