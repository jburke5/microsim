# Michigan Cardiovascular Model (MCM)

This is the broad framework for representing a representative population and modeling individual cardiovascular risk factors, outcomes and cognition.

## Development Quickstart
This section assumes you have Python 3.6+ with `pip` installed. You will need to install `pipenv` if you haven't already:
```
pip install -U pipenv
```

First, clone this GitHub repository using your method of choice. Then, to create a development virtualenv and install development dependencies, run:
```
pipenv install --dev
```

This project uses `flake8` for linting, `autopep8` for formatting, and `unittest` for testing. Each of these tools have Pipenv scripts for convenience:
```
pipenv run lint  # check for flake8 errors
pipenv run format  # format with autopep8
pipenv run format-diff  # what-if for `pipenv run format`
pipenv run test  # run tests
```
