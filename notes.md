# Notes

These notes are a reminder on how to set up a python package using _Visual Studio Code_, with or without a virtual environment, including **github** synchronisation, **readthedocs** documentation using `sphinx`, and **pypi** publishing using `build` and `twine`.

## Virtual Environment

On _Visual Studio Code_, use **Ctrl+Shift+P**, and select _Python: Create Environment_.

Using the terminal (note that this might not work in _Anaconda_ installations, because _Python_ is not automatically added to `%PATH%`):
```
# Use module venv to create the environment .venv:
python -m venv .venv

# List pip packages to check if .venv is activated:
pip list

# Activate the virtual environment:
.venv/Scripts/activate

# Check again:
pip list

# Create requirements.txt:
pip freeze > requirements.txt

# Install from requirements.txt:
pip install -r requirements.txt

# Deactivate:
deactivate
```

## File Structure

A sample file structure is available [here](https://github.com/pypa/sampleproject).

## Github

Syncronization with _Github_ is best managed with an integrated editor, like _Visual Studio Code_.

It is useful to add `README.md` and `LICENSE`. These can be added on _Github_.

On the readme file, it is possible to add badges using [shields.io](https://shields.io/).

## Readthedocs

First, you need to install `sphinx`, including the _readthedocs_ theme:
```
pip install sphinx sphinx_rtd_theme
```

Next, you need to initialize the documentation:
```
md docs
cd docs
sphinx-quickstart
```

Under `conf.py`, check the path, change the theme to `sphinx_rtd_theme` and add the following extensions:
```
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
```

It is possible to generate `.rst` files using:
```
# Must be run under /docs
# Outputs rst files at .
# Read module at ../src
sphinx-apidoc -o . ../src
```

You can then edit the `.rst` files and compile the docs using:
```
# Must be run under /docs
make html
```

You can go to [readthedocs](https://readthedocs.org/) to create or log in to your account, to import the project and documentation from _github_. To build, you might need to add a `requirements.txt` file in the `docs` folder, including the sphinx theme and any imported modules used in the documentation.

Once connected with _github_, _readthedocs_ should update automatically.

## Publishing

First, you need to install `build`
```
pip install build
```

Then, create `pyproject.toml` using the sample file available [here](https://github.com/pypa/sampleproject).

You can build using:
```
python -m build
```

To upload the package you need to install `twine`:
```
pip install twine
```

Then, you can create an account on [TestPyPi](https://test.pypi.org/) or [PyPi](https://pypi.org/), follow the instructions to set your profile and authentification, create a token, save it into a `.pypirc` file on your user directory, and finally upload your package using:
```
# For testpypi:
twine upload --repository testpypi dist/*

# For pypi:
twine upload dist/*
# For updates:
twine upload --skip-existing dist/*
```