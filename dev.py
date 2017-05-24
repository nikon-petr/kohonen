import numpy as np

from lib.hexagon import plot_map

if __name__ == '__main__':
    grid = {'centers': np.array([[1.5, 0.8660254],
                                 [2.5, 0.8660254],
                                 [3.5, 0.8660254],
                                 [1., 1.73205081],
                                 [2., 1.73205081],
                                 [3., 1.73205081],
                                 [1.5, 2.59807621],
                                 [2.5, 2.59807621],
                                 [3.5, 2.59807621]]),
            'x': np.array([3.]),
            'y': np.array([3.])}
    d_m = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ax = plot_map(grid, d_m)
