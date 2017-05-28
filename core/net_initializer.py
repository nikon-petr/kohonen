import numpy as np

from core.net_errors import NetConfigIndefined, IncorrectFactorValue


def initialize(net_object, factor=0.01, negative=True):
    if net_object.config is None:
        raise NetConfigIndefined()
    if abs(factor) > 1:
        raise IncorrectFactorValue()

    net_object.net = []

    net_object.net.append({
        'w': np.random.uniform(-factor if negative else 0, factor, (net_object.config[-1], net_object.config[0])),
        'o': np.zeros((net_object.config[-1])),
    })
