from typing import Iterable

import numpy as np

from utils_ml.src.protocols.Hashed import Hashed


def convert_to_one_hot_vectors(x: Iterable[Hashed]):
    unique_values = list(set(x))

    dict_of_one_hot_vectors = \
        get_dict_convert_to_one_hot_vectors(unique_values)

    one_hot_vectors = list(map(lambda m: dict_of_one_hot_vectors[m], x))

    if type(x) is np.ndarray:
        one_hot_vectors = np.asarray(one_hot_vectors)

    return one_hot_vectors


def get_dict_convert_to_one_hot_vectors(list_of_labels):
    if len(set(list_of_labels)) != len(list_of_labels):
        raise ValueError("All labels must be different")

    list_of_labels.sort()
    dict_of_one_hot_vectors = {}
    for idx, value in enumerate(list_of_labels):
        t = np.zeros(len(list_of_labels))
        t[idx] = 1
        dict_of_one_hot_vectors[value] = t

    return dict_of_one_hot_vectors
