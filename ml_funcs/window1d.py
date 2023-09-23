import numpy as np


def window1d(
    input_array: list | np.ndarray, size: int, shift: int = 1, stride: int = 1
) -> list[list | np.ndarray]:
    if type(input_array) == list():
        input_array = np.array(input_array)
