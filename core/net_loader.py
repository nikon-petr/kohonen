import os
from json import JSONDecodeError, dump
from json import load

import numpy as np

from core.net_errors import JsonFileNotFound, JsonFileStructureIncorrect


def upload(net_object, path):
    if not os.path.isfile(path):
        raise JsonFileNotFound()

    try:
        with open(path, 'r') as file:
            deserialized_file = load(file)
            net_object.config = deserialized_file['config']
            net_object.net = deserialized_file['net']

            if net_object.net:
                net_object.net[-1]['w'] = np.array(net_object.net[-1]['w'])
                net_object.net[-1]['o'] = np.zeros((net_object.config[1]))

    except KeyError:
        raise JsonFileStructureIncorrect()
    except JSONDecodeError:
        raise


def unload(net_object, path):
    try:
        net_copy = [{'w': net_object.net[-1]['w'].tolist()}]

        with open(path, 'w') as file:
            file_dictionary = {
                'config': net_object.config,
                'net': net_copy
            }
            dump(file_dictionary, file, sort_keys=True, indent=4)
    except JSONDecodeError:
        raise
