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

From project root and inside the virtual environment, execute the following commands:

1. Install python package

   ```commandline
   python setup.py install
   ```
2. Generate static documentation

    ```
    sphinx-build -b html docs/source/ docs/build/html
    ```

Afterwards, opening in any browser the local file:
file:///{project-dir}/honcaml/docs/build/html/index.html,
replacing `project-dir` by current project directory, should be enough to see
the documentation.

## Tests
Comanda simple (correr un fitxer de tests):
```
python -m pytest src/tests/data/transform.py
```

Comanda per correr una carpeta de tests:
```
python -m pytest src/tests/data/*
```

Comanda per correr totes les carpetes de tests:
```
python -m pytest src/tests/data/* src/tests/models/* src/tests/steps/* src/tests/tools/*
```

Comanda per correr totes les carpetes de tests + coverage:
```
python -m pytest --cov=src src/tests/data/* src/tests/models/* src/tests/steps/* src/tests/tools/*
```

Comanda per correr totes les carpetes de tests + coverage + missing lines:
```
python -m pytest --cov=src --cov-report term-missing src/tests/data/* src/tests/models/* src/tests/steps/* src/tests/tools/*
```
