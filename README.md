# pyDySP

Dynamic Signal Processing

![GitHub License](https://img.shields.io/github/license/dkaramitros/python-dysp)
[![Read the Docs](https://img.shields.io/readthedocs/pydysp)](https://pydysp.readthedocs.io/en/latest/)
[![PyPI - Status](https://img.shields.io/pypi/status/pydysp)](https://pypi.org/project/pyDySP/)
[![PyPI - Version](https://img.shields.io/pypi/v/pydysp)](https://pypi.org/project/pyDySP/)

## Information

Python classes are provided to process laboratory data from the Shaking Table experimental facilities at the University of Bristol.

## Installation

### Using pip

PyDySP is available via `pip` and can be installed with:
```
pip install pydysp
```

If you are using a virtual environment and want to also use a _Jupyter_ notebook, make sure you also install `ipykernel` with:
```
pip install ipykernel
```

### Github clone

You can clone the source code using:
```
git clone https://github.com/dkaramitros/pyDySP
```

### Manual download

You can also find the source code under [releases](https://github.com/dkaramitros/pyDySP/releases), including example jupyter notebooks. You can download the code manually, extract in your working subfolder, and use it directly.

## Instructions

The package `pyDySP` includes two classes:

- The `Test` class provides methods to add, manage, and plot data from multiple channels. Transfer functions can also be produced and analyzed.

- The `Channel` class provides methods for signal processing, including baseline correction, filtering, and trimming. Plotting methods are also provided.

Detailed documentation on the available class methods is available [here](https://pydysp.readthedocs.io/)

To help with the use of the software, Example Jupyter notebooks are also provided [here](https://github.com/dkaramitros/pyDySP/tree/main/examples).