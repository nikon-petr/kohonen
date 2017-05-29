import math
from abc import ABCMeta, abstractmethod

from core.net_errors import NetIsNotInitialized, NetIsNotCalculated


class Corrector:
    __metaclass__ = ABCMeta

    def __init__(self, nu, tau):
        self.__nu = nu
        self.__tau = tau
        self._t = 0

    def nu(self):
        return self.__nu * math.exp(-self._t / self.__tau)

    def sigma(self):
        return self.__nu * math.exp(-self._t / self.__tau)

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
