import math

import numpy as np

from core.net_abstract_corrector import Corrector


class Standard(Corrector):
    def __init__(self, nu=1, tau=1000):
        super(Standard, self).__init__(nu, tau)

    def initialize(self, net_object):
        if net_object.net[-1].get('p') is None:
            net_object.net[-1]['p'] = np.full((net_object.config[-1]), 1 / net_object.config[-1])

    def correct(self, net_object):
        super(Standard, self).correct(net_object)

        net = net_object.net

        bmu_index = net[-1]['o'].argmin(axis=0)

        d = np.apply_along_axis(lambda ij: math.sqrt(np.sum((net[-1]['i'][bmu_index] - ij) ** 2)), 1, net[-1]['i'])

        h = np.vectorize(math.exp)(-d / self.sigma())

        w = np.apply_along_axis(lambda x: x - net[0]['o'], 1, net[-1]['w'])

        wh = np.apply_along_axis(lambda x: x * h, 0, w)

        g = wh * self.nu()

        net[-1]['w'] -= g

        del net[0]
