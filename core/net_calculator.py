import pprint

import numpy as np

from core.net_errors import IncorrectInputVectorLength


def calculate(net_object, input_vector, training=False):
    if net_object.config[0] != len(input_vector):
        raise IncorrectInputVectorLength()

    net = net_object.net

    net.insert(0, {'o': input_vector})

    net[-1]['o'] = np.apply_along_axis(lambda x: net_object.d(input_vector, x), 1, net[-1]['w'])

    if not training:
        del net[0]

    net_object.is_calculated = True
