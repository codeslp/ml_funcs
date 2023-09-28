import pytest
import sys
sys.path.append('../')
import numpy as np
from ml_functions.window1d import window1d
from ml_functions.convolution2d import convolution2d
from ml_functions.transpose2d import transpose2d

def test_window1d_with_list():
    input_data = [1, 2, 3, 4, 5, 6]
    expected_output = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    result = window1d(input_data, 2)
    assert result == expected_output, f'Expected {expected_output}, but got {result}'

def test_window1d_with_np_array():
    input_data = np.array([1, 2, 3, 4, 5, 6])
    expected_output = np.array([[1, 2], [3, 4], [5, 6]])
    result = window1d(input_data, 2, 2)
    np.testing.assert_array_equal(result, expected_output)

def test_transpose2d():
    input_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    expected_output = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
    result = transpose2d(input_matrix)
    assert result == expected_output, f'Expected {expected_output}, but got {result}'


def test_convolution2d():
    input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 0], [0, -1]])
    expected_output = np.array([[-4., -4.], [-4., -4.]]) 
    result = convolution2d(input_matrix, kernel)
    np.testing.assert_array_equal(result, expected_output)