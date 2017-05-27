import pprint


if __name__ == '__main__':
    from core.net_interface import Net
    from core.net_standard_corrector import Standard
    from core.net_potential_corrector import Potential
    from core.distance_functions import distance_functions
    from dataset.dataset import dataset

    # distances = np.random.uniform(0, 1, (20, 20))
    # save_image(distances)

    net = Net(distance_functions['euclidean'], Potential)

    dataset = dataset('/Users/nikon/PycharmProjects/lakohonen/data/iris.train.csv')

    net.initialize(4, 40, 40, 0.1)

    net.train(30, dataset)

    net.save_to('/Users/nikon/PycharmProjects/lakohonen/data/iris.net.json')
