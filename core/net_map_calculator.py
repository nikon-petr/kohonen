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

    dataset_indexes = [[] for i in range(net_object.config[-1])]

    for count, vector in enumerate(dataset):
        net.insert(0, {'o': vector})
        net[-1]['o'] = np.apply_along_axis(lambda x: net_object.d(vector, x), 1, net[-1]['w'])

        j = net[-1]['o'].argmin(axis=0)

        dataset_indexes[j].append(count)

    mean_parameter = []
    for count, indexes in enumerate(dataset_indexes):
        if not indexes:
            mean_parameter.append(net[-1]['w'][count])
        else:
            mean_parameter.append(f[mode](dataset[indexes], axis=0))

    mean_parameter = np.array(mean_parameter).T
    mean_parameter = mean_parameter.reshape((mean_parameter.shape[0], net_object.m, net_object.n))
    return mean_parameter
