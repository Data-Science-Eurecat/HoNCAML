# Contributing guide

If you feel like contributing to HoNCAML, there are multiple ways to do it:

- Pick a planned development/feature from [our list](https://github.com/Data-Science-Eurecat/HoNCAML/blob/main/TODO.md)
- Fix reported issues
- Improve documentation

## Installation

In order to contribute, it is just a matter of cloning the repository, and
start hacking. It is recommended to do so within a virtual environment, so that
is why there is a specific make target to ease the set up: `make all`, which
does the following:

- Create virtual environment through venv module (requires *python3-venv*
  module)
- Installs honcaml in development mode, so that code changes are reflected live
  in the library

## Coding style

For coding style, we try to ensure the following:

- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use of [type hints](https://peps.python.org/pep-0484/) in functions
- Use of [this docstring
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Required checks

Before sending any patch, please ensure the following:

- Code style has no problems: `make validate_code`
- All tests are passed: `make tests`
- All example pipelines from **honcaml/config/examples** run without problems
- All documentation is up to date with the changes
    - Sphinx documentation
    - Internal documentation
        - README.md
        - TODO.md
        - Template configuration files from **honcaml/config/templates**

In order to perform this checks, additional dependencies will be needed. These
can be installed by `pip install honcaml[check] honcaml[document]
honcaml[tests]`.

It is not compulsory, but clearly encouraged, to create tests to validate
any new functionality that has been introduced.

## Submitting patches

When development has finished and checks have been performed, code changes can
be submitted creating a pull request.

Please ensure that the title is clear and the description provides all the
necessary explanations to understand the motivation and the context.
