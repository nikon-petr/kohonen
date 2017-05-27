import math
from abc import ABCMeta, abstractmethod

from core.net_errors import NetIsNotInitialized, NetIsNotCalculated


class Corrector:
    __metaclass__ = ABCMeta

    def __init__(self, nu, tau):
        self._nu = lambda: nu * math.exp(-self._t / tau)
        self._sigma = lambda: nu * math.exp(-self._t / tau)
        self._t = 0

    @abstractmethod
    def initialize(self, net_object):
        pass

    @abstractmethod
    def correct(self, net_object):
        if net_object.net is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        self.initialize(net_object)

        self._t += 1
