import pprint

import numpy as np

from core.net_errors import NetIsNotInitialized, UnknownAggregationFunction


def calculate_map(net_object, dataset, mode='avg'):
    f = {
        'avg': np.average,
        'max': np.max,
        'min': np.min
    }

    if net_object.net is None:
        raise NetIsNotInitialized()
    if mode not in f:
        raise UnknownAggregationFunction()

    net = net_object.net

    v = [[] for i in range(net_object.config[-1])]

    for count, vector in enumerate(dataset):
        net.insert(0, {'o': vector})
        net[-1]['o'] = np.apply_along_axis(lambda x: net_object.d(vector, x), 1, net[-1]['w'])

        j = net[-1]['o'].argmin(axis=0)

        v[j].append(count)

    mean = []
    for count, i in enumerate(v):
        if len(i) == 0:
            mean.append(net[-1]['w'][count])
        else:
            mean.append(f[mode](dataset[i], axis=0))

    m_mean = np.array(mean).T
    m_mean = m_mean.reshape((m_mean.shape[0], net_object.m, net_object.n))
    return m_mean
