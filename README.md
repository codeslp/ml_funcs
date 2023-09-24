# ml_funcs: A Machine Learning Functions Package

## Overview

The `ml_funcs` package provides a collection of Python functions designed for various machine learning tasks. The package currently includes:

- `convolution2d`: Perform 2D cross-correlation between an input matrix and a kernel.
- `transpose`: Transpose a 2D matrix.
- `window1d`: Generate a list of windows from a 1D array or list.

## Installation

This package uses Poetry for dependency management. To install the package, first, ensure you have [Poetry installed](https://python-poetry.org/docs/#installation).

Then, run:

```bash
poetry install
```

## Usage

### convolution2d

Perform 2D cross-correlation.

```python
from ml_funcs import convolution2d
result = convolution2d(input_matrix, kernel)
```


### transpose

Transposes a 2D matrix.

```python
from ml_funcs import transpose
result = transpose(input_matrix)
```


### window1d

Generate a list of windows from a 1D array or list.

```python
from ml_funcs import window1d
result = window1d(input_array, size)
```


