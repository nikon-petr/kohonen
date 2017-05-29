import pprint

import numpy as np

from core.net_errors import NetIsNotInitialized, UnknownAggregationFunction


def calculate_clusters(net_object, dataset, classes):
    if net_object.net is None:
        raise NetIsNotInitialized()

    net = net_object.net

    dataset_indexes = [[] for i in range(net_object.config[-1])]

    for count, vector in enumerate(dataset):
        net.insert(0, {'o': vector})
        net[-1]['o'] = np.apply_along_axis(lambda x: net_object.d(vector, x), 1, net[-1]['w'])

        j = net[-1]['o'].argmin(axis=0)

        dataset_indexes[j].append(count)

    cluster = [classes[i[0]] if i else -1 for i in dataset_indexes]
    return cluster
