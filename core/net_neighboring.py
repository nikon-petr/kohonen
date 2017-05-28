import numpy as np

from core.net_errors import NetIsNotInitialized


def calculate_average_neighboring(net_object, m, n):
    if net_object.net is None:
        raise NetIsNotInitialized()

    net = net_object.net

    zero_weights = np.zeros((net_object.config[0]))

    weights = np.ma.array(np.reshape(net[-1]['w'], (m, n, zero_weights.shape[0])), mask=False)
    weights = np.insert(weights, (0, weights.shape[1]), 0, axis=1)
    weights = np.insert(weights, (0, weights.shape[0]), 0, axis=0)
    weights.mask = True
    weights.mask[1:-1, 1:-1] = False

    result = np.zeros((m, n))
    for i, j in np.ndindex(weights.shape[:2]):
        if not weights.mask[i, j].all():
            a = [[i - 1, i - 1, i, i, i + 1, i + 1], [j - 1, j, j - 1, j + 1, j - 1, j]]
            d = np.apply_along_axis(lambda v: net_object.d(weights[i, j], v), 1, weights[a])
            result[i - 1, j - 1] = np.nanmean(d)

    return result
