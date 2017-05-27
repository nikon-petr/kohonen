import numpy as np
from math import sqrt

normalization_functions = {
    'normalization_1': lambda x: x / sqrt(np.sum(x ** 2)),
    'normalization_2': lambda x: x / np.abs(x)
}