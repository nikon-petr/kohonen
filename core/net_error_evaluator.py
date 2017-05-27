import numpy as np

from core.net_errors import NetIsNotInitialized, NetIsNotCalculated


def evaluate(net_object, input_vector):
    if net_object.net is None:
        raise NetIsNotInitialized()
    if not net_object.is_calculated:
        raise NetIsNotCalculated()

    net = net_object.net

    winner_index = net[-1]['o'].argmin(axis=0)
    e = np.mean((input_vector - net[-1]['w'][winner_index, :]) ** 2)

    return e
