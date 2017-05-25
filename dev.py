import numpy as np

if __name__ == '__main__':
    from core.net_visualizer import save_image

    distances = np.random.uniform(0, 1, (4, 5))
    save_image(distances)
