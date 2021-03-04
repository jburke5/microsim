# MICROSIM: Michigan Chronic Disease Simulation

This is the broad framework for representing a representative population and modeling individual cardiovascular risk factors, outcomes and cognition.

## Development Quickstart
This section assumes you have Python 3.7+ with `pip` installed. You will also need to [install `poetry`](https://poetry.eustace.io/docs/#installation) to install dependencies, run development commands, and build the package.

For a more gentle introduction, try the [installation guide](INSTALL.md).

First, clone this GitHub repository using your method of choice. Then, to create a development virtualenv and install development dependencies, run:
```
poetry install
```

This project uses `flake8` for linting, `black` for formatting, and `unittest` for testing. Each of these tools have Poetry scripts for convenience:
```
poetry run lint  # check for flake8 errors
poetry run format  # format with black
poetry run test  # run tests
```
