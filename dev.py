import pprint


if __name__ == '__main__':
    from core.net_interface import Net
    from core.net_standard_corrector import Standard
    from core.net_potential_corrector import Potential
    from core.distance_functions import distance_functions
    from dataset.dataset import dataset

    # distances = np.random.uniform(0, 1, (20, 20))
    # save_image(distances, '/Users/nikon/PycharmProjects/lakohonen/data/map.png')

    net = Net(distance_functions['euclidean'], Potential)

    dataset = dataset('/Users/nikon/PycharmProjects/lakohonen/data/iris.train.csv')

    net.initialize(input=4, m=4, n=4, factor=1, negative=False)
    # net.load_from('/Users/nikon/PycharmProjects/lakohonen/data/iris.net.json')

    net.train(100, dataset, stop_error=10 ** -20, stop_delta=10 ** -30)

    net.visualize_maps(dataset, 'max', '/Users/nikon/PycharmProjects/lakohonen/data')
    net.visualize_u_matrix('/Users/nikon/PycharmProjects/lakohonen/data/u-matrix.png')

    net.save_to('/Users/nikon/PycharmProjects/lakohonen/data/iris.net.json')
