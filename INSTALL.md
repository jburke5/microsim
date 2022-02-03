# Installation guide for beginners

**Table of Contents**

- [Installation guide for beginners](#installation-guide-for-beginners)
    - [Concepts](#concepts)
    - [Guide](#guide)


## Concepts

The MICROSIM project is built on many other software packages that serve the following roles:

* **Development environment with statistical tools**: We use the [Python programming language](https://www.python.org/) with the [statsmodels module](https://www.statsmodels.org/stable/index.html) to run the statistical modeling and analysis. *This environment is what actually runs the code*, roughly analogous to Stata/SPSS. The term "environment" is used because it contains many tools that work together to make it possible. Below are the main players and their roles:
  * Python 3.7 - a general purpose programming language that has a large software ecosystem
  * statsmodels - statistical modeling, testing, and data exploration
  * numpy, scipy, pandas - basic packages for numerical analysis, scientific functions, and data frames
  * [Poetry](https://python-poetry.org/) - tool to install Python packages such as those listed above

* **Collaboration**: We use [git](https://git-scm.com/) and [git-lfs](https://git-lfs.github.com/) to collaborate on this project to share the code and data. These are known as "version management tools" because they save historical versions, similar to FinalPaper-2020-05-17.pdf. Generally speaking, git is used for code, and git-lfs is used for large data files. These tools are used to sync to [GitHub](https://github.com/jburke5/microsim).

* **Editing**: We recommend that you use [VS Code](https://code.visualstudio.com/), a free and open-source text editor by Microsoft that has easily installed extensions for git, Python, and much more.

## Guide

Installing all of the above can be daunting, but the steps below should give a step-by-step tutorial of how to get up and running. Some steps can vary considerably between operating systems.

**Windows:** Many of these tools work best in the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or you may need to use slightly modified tools (ex: [pyenv-win](https://github.com/pyenv-win/pyenv-win)).

**macOS:** [homebrew](https://brew.sh/) is a popular command line tool to install development tools. If you anticipate needing to install more technical packages long term or already have it installed, it is preferable to install brew and then install pyenv, pipx, GitHub Desktop, and VS Code with it.

**Linux:** Many of these tools are available through your system package manager. You may wish to install these packages through there if they are available if it makes system maintenance easier, but the instructions below should work as expected by installing everything to your user.

1. Install [pyenv](https://github.com/pyenv/pyenv#installation), a tool to automatically install Python. To do this, open a Terminal (macOS: open Spotlight, search for Terminal), and paste the following code line by line and run each line by pressing <kbd>ENTER</kbd>.

    ```
    curl https://pyenv.run | bash
    exec $SHELL
    ```

1. Install Python 3.7 with pyenv. On macOS and Linux, running the following command will install 3.7.7, the latest release of Python 3.7:

    ```
    pyenv install 3.7.7
    pyenv global 3.7.7
    ```

1. Install [pipx](https://github.com/pipxproject/pipx), a tool to manage command line Python tools

    ```
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

1. Install [poetry](https://python-poetry.org/docs/#installation) using pipx:

    ```
    pipx install poetry
    ```

1. Install [GitHub desktop](https://desktop.github.com/)

1. Create a [GitHub account](https://github.com/) and use that to log into the GitHub desktop.

1. Follow these [instructions](https://help.github.com/en/desktop/getting-started-with-github-desktop/configuring-git-for-github-desktop) on setting up GitHub desktop with your name and email.

1. Follow these [instructions](https://help.github.com/en/desktop/contributing-to-projects/cloning-a-repository-from-github-to-github-desktop) with the [MICROSIM repository](https://github.com/jburke5/microsim).
**Tip:** "clone" roughly means "download with version history", and "repository" roughly means "project folder".

1. Install [VS Code](https://code.visualstudio.com/Download). Once it is installed, open it, open the project folder, and install the Python extension when offered.

1. Within VS Code, you can open a Terminal window, which should open in your project folder. Once you are there, you can run the `poetry install` command as in the README.

    ```
    poetry install
    ```

1. Your installation should be complete at this point, but to make sure that everything is working as expected, run the following command to run the automated tests:

    ```
    poetry run ./scripts/test.py
    ```

1. Congratulations! You made it!
