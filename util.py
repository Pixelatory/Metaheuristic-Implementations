import random

import numpy as np


def gen_random_vector_idx(size):
    """
    Generates a random index vector which doesn't allow for
    the same value as the index it's placed in.

    i.e. doesn't allow for [0, 1, 2, 3, ...]
    or [8, 1, 8, 8, 8]
    or [5, 6, 7, 3, 10]

    :type size: int
    """
    res = []
    for i in range(size):
        res.append(random.sample([j for j in range(size) if j != i], 1)[0])
    return res


def lehmer_mean(data):
    data_sum = np.sum(data)
    if data_sum == 0:
        return 0
    return np.sum(np.power(data, 2)) / data_sum