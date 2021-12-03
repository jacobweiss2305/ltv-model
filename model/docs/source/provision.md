# Provisioning Process

## Python Versioning

We use pyenv to manage our python versioning.

### Pyenv Installation

- Windows installation: https://github.com/pyenv-win/pyenv-win
- Linux/Mac installation: https://github.com/pyenv/pyenv

#### Install Pyenv on Mac

1. Install [brew](https://brew.sh/) on Mac

2. Install pyenv using brew

```sh
brew install pyenv
```

3. Install xz

```sh
brew install xz
```

4. add to ~/.zprofile

```sh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init --path)"
```

5. add to ~/.zshrc

```sh
eval "$(pyenv init -)"
```

#### Install Pyenv on Debian/Ubuntu

1. Install pyenv dependencies

```sh
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

2. Install pyenv

```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

3. add to ~/.zshrc

```sh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```

### Pyenv Activation

- Activate Pyenv (this reads from .python-version)

```
pyenv install
```

- If the python version is not compatible to your machine, try to install a compatible python version with the same major version, such as 3.8.9 and 3.8.11 are all under 3.8

```
pyenv install 3.8.9
```

## Python Virtual Environments

We use venv to manage our python virtual environments.

1. Create virtual environment

   - for dev

     ```
     python -m venv venv
     ```

   - for prod
     ```
      python -m venv prod_env
     ```

2. Activate Virtual Environment

   - for dev

     - for Windows

       ```
       venv\Scripts\activate.bat
       ```

     - for Linux/Mac

       ```
       source venv/bin/activate
       ```

   - for prod

     - for Windows

       ```
       prod_env\Scripts\activate.bat
       ```

     - for Linux/Mac

       ```
       source prod_env/bin/activate
       ```

3. Verify virtual environment

```
which pip
# should be in the `venv/bin` for dev or `prod_env/bin` for prod
```

4. Install pip packages

   - (remove the pywin package in requirements.txt if your os is not windows before running the command)
   - for dev
     ```
     pip install -r requirements.txt
     ```
   - for prod
     ```
     pip install -r prod_requirements.txt
     ```

## Pre-commit formatting

These are automatic checks of formatting issues in files.

Currently includes: trailing whitespace, end of file space, black, isort.

- Install pre-commit package

```
$ pre-commit install
```
