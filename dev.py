import numpy as np

if __name__ == '__main__':
    from core.visualizer import save_image

    distanses = np.random.uniform(0, 1, (9, 9))
    save_image(distanses)
