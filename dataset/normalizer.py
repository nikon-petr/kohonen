import math
import numpy as np


def normalize_dataset(dataset):
    normalized = []
    for vector in dataset:
        normalized.append(np.vectorize(lambda x: x / math.sqrt(np.sum(vector ** 2)))(vector))
    return np.array(normalized)
