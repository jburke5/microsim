# Michigan Cardiovascular Model (MCM)

This is the broad framework for representing a representative population and modeling individual cardiovascular risk factors, outcomes and cognition.

## Development Quickstart
This section assumes you have Python 3.7+ with `pip` installed. You will also need to [install `poetry`](https://poetry.eustace.io/docs/#installation) to install dependencies, to run development commands, and to build the package.

First, clone this GitHub repository using your method of choice. Then, to create a development virtualenv and install development dependencies, run:
```
poetry install
```

This project uses `flake8` for linting, `autopep8` for formatting, and `unittest` for testing. Each of these tools have Poetry scripts for convenience:
```
poetry run lint  # check for flake8 errors
poetry run format  # format with autopep8
poetry run format-diff  # what-if for `poetry run format`
poetry run test  # run tests
```
