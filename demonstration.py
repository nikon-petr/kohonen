import pprint


if __name__ == '__main__':
    from core.net_interface import Net
    from core.net_standard_corrector import Standard
    from core.net_potential_corrector import Potential
    from core.net_distance_functions import distance_functions
    from dataset.dataset import dataset

    dataset = dataset('/Users/nikon/PycharmProjects/lakohonen/data/iris.train.csv')
    classes = [j for j in range(3) for i in range(50)]

    potential = Potential(nu=1, tau=3000, p_min=0.75)
    net = Net('Iris', distance_functions['manhattan'], potential)
    net.initialize(input=4, m=10, n=15, factor=1, negative=False)
    # net.load_from('/Users/nikon/PycharmProjects/lakohonen/data/iris.cluster.net.json')
    net.train(500, dataset, stop_error=10 ** -15, stop_delta=10 ** -15)

    path = '/Users/nikon/PycharmProjects/lakohonen/data'
    net.visualize_maps(dataset, 'avg', path)
    net.visualize_u_matrix(path)
    net.visualize_clusterization(dataset, cluster_number=3, classes=classes, path=path)

    net.save_to('/Users/nikon/PycharmProjects/lakohonen/data/iris.net.json')
