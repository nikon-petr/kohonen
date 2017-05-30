import math

import numpy as np

distance_functions = {
    'euclidean': lambda x1, x2: math.sqrt(np.sum((x1 - x2) ** 2)),
    'euclidean_sqr': lambda x1, x2: np.ma.sum((x1 - x2) ** 2),
    'manhattan': lambda x1, x2: np.ma.sum(np.abs(x1 - x2))
}