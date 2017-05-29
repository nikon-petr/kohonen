import math

import numpy as np

from core.net_abstract_corrector import Corrector


class Potential(Corrector):
    def __init__(self, nu=0.1, tau=1000, p_min=0.9):
        super(Potential, self).__init__(nu, tau)
        self.__p_min = p_min

    def initialize(self, net_object):
        if net_object.net[-1].get('p') is None:
            net_object.net[-1]['p'] = np.full((net_object.config[-1]), self.__p_min)

    def correct(self, net_object):
        super(Potential, self).correct(net_object)

        net = net_object.net

        net_output = np.ma.array(net_object.net[-1]['o'], mask=False)

        inactive = np.argwhere(net[-1]['p'] < self.__p_min)
        net_output.mask[inactive] = True

        winner_index = net_output.argmin(axis=0)

        # print(winner_index)

        d = np.apply_along_axis(lambda x: net_object.d(net[-1]['w'][winner_index], x), 1, net[-1]['w'])

        h = np.vectorize(math.exp)(-d / (2 * (self._sigma() ** 2)))

        w = np.apply_along_axis(lambda x: x - net[0]['o'], 1, net[-1]['w'])

        wh = np.apply_along_axis(lambda x: x * h, 0, w)

        g = wh * self._nu()

        net[-1]['w'] -= g

        net[-1]['p'] += 1 / len(net[-1]['p'])
        net[-1]['p'][winner_index] -= (self.__p_min + 1 / len(net[-1]['p']))

        del net[0]
