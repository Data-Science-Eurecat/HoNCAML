# Contributing guide

If you feel like contributing to HoNCAML, there are multiple ways to do it:

- Pick a planned development/feature from [our list](TODO.md)
- Fix reported issues
- Improve documentation

## Coding style

For coding style, we try to ensure the following:

- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use of [type hints](https://peps.python.org/pep-0484/) in functions
- Use of [this docstring
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Required checks

Before sending any patch, please ensure the following:

- Code style has no problems: `make validate_code` (require flake8 package)
- All tests are passed: `make tests` (require pytest and pytest-cov packages)
- Pipelines run without problems:
    - Train
    - Predict
    - Benchmark
- All documentation is up to date with the changes
    - Sphinx documentation
    - Internal documentation
        - README.md
        - TODO.md
        - Configuration files from [examples](honcaml/config/examples)
        - Configuration files from [templates](honcaml/config/templates)

It is not compulsory, but clearly encouraged, to create tests to validate
any new functionality that has been introduced.

## Submitting patches

When development has finished and checks have been performed, code changes can
be submitted creating a pull request.

Please ensure that the title is clear and the description provides all the
necessary explanations to understand the motivation and the context.
