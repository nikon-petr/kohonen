import random
import os
import numpy as np

from core.net_calculator import calculate
from core.net_cluster_calculator import calculate_clusters
from core.net_error_evaluator import evaluate
from core.net_initializer import initialize
from core.net_map_calculator import calculate_map
from core.net_visualizer import make_image
from core.net_neighboring_calculator import calculate_average_neighboring
from core.net_loader import upload, unload
from core.net_state import NetState
from lib.colors import Colors


def raise_exceptions(f):
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except:
            raise

    return wrapper


class Net:
    def __init__(self, name, f, corrector):
        self.__state = NetState(f, name)
        self.__corrector = corrector

    @property
    def corrector(self):
        return self.__corrector

    @corrector.setter
    def corrector(self, new_corrector):
        self.__corrector = new_corrector

    @raise_exceptions
    def load_from(self, path):
        upload(self.__state, path)

    @raise_exceptions
    def save_to(self, path):
        unload(self.__state, path)

    @raise_exceptions
    def initialize(self, input, m, n, factor, negative):
        self.__state.m = m
        self.__state.n = n
        self.__state.config = [input, m * n]
        initialize(self.__state, factor, negative)

    @raise_exceptions
    def calculate(self, vector):
        calculate(self.__state, vector)
        print(evaluate(self.__state, vector))

    @raise_exceptions
    def train(self, epoch, train_data, stop_error=None, stop_delta=None):
        print("TRAINING STARTED")
        em = 0
        for epoch in range(epoch):
            e_sum = 0
            train_data_indexes = random.sample(range(len(train_data)), len(train_data))
            for d in train_data_indexes:
                calculate(self.__state, train_data[d], training=True)
                e_sum += evaluate(self.__state, train_data[d])
                self.__corrector.correct(self.__state)

            delta = em - e_sum / len(train_data)
            em = e_sum / len(train_data)
            em_color = Colors.OKGREEN if em < 0.1 else Colors.FAIL
            d_color = Colors.OKGREEN if delta > 0 else Colors.FAIL

            print('EPOCH:%s %sEm = %e%s\t %sD = %e%s' % (
            epoch, em_color, em, Colors.ENDC, d_color, delta, Colors.ENDC))

            if (stop_delta and abs(delta) < stop_delta) or (stop_error and em < stop_error):
                print("TRAINING STOPPED")
                break

    def visualize_u_matrix(self, path):
        distances = calculate_average_neighboring(self.__state)
        title = '%s U-matrix %sx%s' % (self.__state.name, self.__state.m, self.__state.n)
        file = os.path.join(path, '%s.umatrix.png' % self.__state.name)
        make_image(distances, file, title=title, color_map='binary', axis_caption='Mean distance')

    def visualize_maps(self, dataset, mode, path):
        mean = calculate_map(self.__state, dataset, mode)
        for n, r in enumerate(mean):
            title = '%s Feature %s %sx%s' % (self.__state.name, n + 1, self.__state.m, self.__state.n)
            file = os.path.join(path, '%s.feature_%s.png' % (self.__state.name, n + 1))
            make_image(r, file, title=title, color_map='inferno', axis_caption='Mean value')

    def visualize_clusterization(self, dataset, cluster_number, classes, path):
        distances = calculate_average_neighboring(self.__state)
        clusters = calculate_clusters(self.__state, dataset, classes)
        title = '%s Clusterization %sx%s' % (self.__state.name, self.__state.m, self.__state.n)
        file = os.path.join(path, '%s.clusterization.png' % self.__state.name)
        make_image(
            values=distances,
            clusters=clusters,
            cluster_number=cluster_number,
            path=file,
            title=title,
            color_map='binary',
            axis_caption='Mean value'
        )
