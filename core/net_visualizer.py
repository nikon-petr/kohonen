import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_image(values, path, clusters=None, cluster_number=None, color_map='inferno', title='Kohonen Map', dpi=100, scale=5.12, axis_caption='Mean', show=False):
    """
    Shows and saves Kohonen map for given distances
    :param axis_caption: Axis caption
    :param show: If it is true shows result map
    :param scale: standard scale of map
    :param dpi: dots per inch
    :param values: values of distances for Kohonen map
    :param color_map: color map e.g. "binary", "Inferno", "Purples", "Magma", "Plasma", "Viridis"
    :param title: title of map
    :return: None
    """

    m = len(values[0])
    n = len(values)

    hexagon_out_r = 0.5
    hexagon_in_r = hexagon_out_r * math.sqrt(3) / 2
    space = 0.04

    odd_row_start = hexagon_in_r
    even_row_start = odd_row_start + hexagon_in_r + space / 2

    odd_row_end = odd_row_start + (2 * hexagon_in_r + space) * (m - 1)
    even_row_end = odd_row_end + hexagon_in_r + space / 2

    col_start = hexagon_out_r
    col_end = col_start + ((2 * hexagon_in_r + space) * math.sqrt(3) / 2) * (n - 1)

    grid_center = (even_row_end + hexagon_in_r) / 2, (col_end + hexagon_out_r) / 2
    centring = (m + 1) / 2 - grid_center[0], (n + 1) / 2 - grid_center[1]

    odd_row_start += centring[0]
    even_row_start += centring[0]
    odd_row_end += centring[0]
    even_row_end += centring[0]
    col_start += centring[1]
    col_end += centring[1]

    odd_x_offsets = np.linspace(odd_row_start, odd_row_end, m)
    even_x_offsets = np.linspace(even_row_start, even_row_end, m)
    y_offsets = np.linspace(col_start, col_end, n)

    odd_xy_offsets = np.array(list(itertools.product(y_offsets[::2], odd_x_offsets)))
    even_xy_offsets = np.array(list(itertools.product(y_offsets[1::2], even_x_offsets)))

    xy_offsets = np.concatenate((odd_xy_offsets, even_xy_offsets))
    xy_offsets = xy_offsets[np.lexsort((xy_offsets[:, 1], xy_offsets[:, 0]))]
    xy_offsets = xy_offsets[:, ::-1]

    fig, ax = plt.subplots(1)

    d = values.flatten()

    cm_object = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
    colorized_distances = cm_object.to_rgba(d)
    hexagon_list = []

    if clusters:
        colors = plt.cm.rainbow(np.linspace(0, 1, 3))
        for i in range(colorized_distances.shape[0]):
            if clusters[i] >= 0:
                colorized_distances[i] = colors[clusters[i]]

    for c, xy in zip(colorized_distances, xy_offsets):
        hexagon_list.append(
            RegularPolygon(
                xy=xy,
                numVertices=6,
                radius=hexagon_out_r,
                orientation=0.,
                facecolor=c
            )
        )

    pc = PatchCollection(hexagon_list, match_original=True)
    ax.add_collection(pc)

    # hack
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    # end hack

    ax.axis([0, m + 1, 0, n + 1])
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label(axis_caption)

    ax.tick_params(
        which='both',
        bottom='off',
        left='off',
        labelbottom='off',
        labelleft='off'
    )

    w = scale * (m / n) if m >= n else scale
    h = scale * (n / m) if m <= n else scale

    fig.set_size_inches(w, h)

    ax.invert_yaxis()

    if show:
        plt.show()

    plt.savefig(path, dpi=dpi)
