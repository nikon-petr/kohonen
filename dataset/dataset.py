import csv

import numpy as np

from dataset.normalizer import normalize_dataset


def dataset(csv_path):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        input_vectors = []
        for row in reader:
            input_vectors.append(row)
        dataset = np.array(input_vectors)
        dataset = normalize_dataset(dataset)
        return dataset